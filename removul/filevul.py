import re
from removul.linelevel import summerize_attention,remove_special_token_score,get_lines_score
from removul.linevul import UCC_Dataset
import torch
import os
import logging
logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)

import re

def is_start_of_function(line):
    """Returns True if the given line is the start of a C or C++ function, False otherwise
            /
        ^                            # Start of line
        \s*(?:struct\s+)[a-z0-9_]+   # return type
        \s*\**                       # return type can be a pointer
        \s*([a-z0-9_]+)              # Function name
        \s*\(                        # Opening parenthesis
        (
            (?:struct\s+)            # Maybe we accept a struct?
            \s*[a-z0-9_]+\**         # Argument type
            \s*(?:[a-z0-9_]+)        # Argument name
            \s*,?                    # Comma to separate the arguments
        )*
        \s*\)                        # Closing parenthesis
        \s*{?                        # Maybe a {
        \s*$                         # End of the line
        /mi                          # Close our regex and mark as case insensitive
    """
    # pattern check line not end with ;
    line = line.strip()
    if line.endswith(';'):
        return False
    line = line.strip()
    pattern = r'^\s*(?:struct\s+)?[a-zA-Z0-9_]+\s*\**\s*([a-zA-Z0-9_]+)\s*\((?:(?:struct\s+)?\s*[a-zA-Z0-9_]+\**\s*[a-zA-Z0-9_]+\s*(?:=\s*[^,]+)?\s*,?\s*)*(char\s*\*\s*[a-zA-Z0-9_]+\s*(?:\[\])?(?:=\s*[^,]+)?)?\)\s*{?'
    return re.match(pattern, line, re.IGNORECASE) is not None


def get_c_and_cpp_files(directory_path):
    """ Returns a list of paths to all the C and C++ files in the given directory and its subdirectories """
    # Initialize an empty list to store file paths
    file_paths = []

    # Loop over all the files and directories in the given path
    for entry in os.scandir(directory_path):
        # Check if the entry is a file
        if entry.is_file():
            # Check if the file is a C or C++ file by its extension
            if entry.name.endswith('.c') or entry.name.endswith('.cpp'):
                # If it is, add its full path to the list of file paths
                file_paths.append(entry.path)
        # Check if the entry is a directory
        elif entry.is_dir():
            # Recursively call the function on the subdirectory and append its results to the list of file paths
            file_paths.extend(get_c_and_cpp_files(entry.path))

    # Return the list of file paths
    return file_paths

# Define a function to read a C file and return its contents as a string
def read_c_file(filename):
    """ Returns the contents of the given C or C++ file as a string """
    with open(filename, 'r') as file:
        code = file.read()
        return code
    
# Define a function to tokenize a C function from a string
def tokenize_c_function(code):
    """ Returns a list of dictionaries, each containing the code for a function in the given C or C++ file
    and a mapping between the line number of the function and the line number of the file
    e.g. [{'function': 'int main() {\n    printf("Hello World!");\n //comment \n   return 0;\n}', 'mapper': {1: 1, 2: 2, 3: 4, 4: 5}}] """
    functions = []
    in_function = False
    function_lines = []
    infile_line=0
    infunction_line=1
    start_comment=False
    # hash table mape between infunction_line and inline
    mapper={}
    for line in code.split('\n'):
        infile_line+=1
        if is_start_of_function(line):
            if in_function:
                functions.append({'function':'\n'.join(function_lines) , 'mapper':mapper})
                function_lines = []
                infunction_line=1
                mapper={}
            in_function = True
        if in_function:
            # remove comment from line this expected to get better line level detaction
            # in vulnarability in the function
            line = re.sub(r"/\*.*?\*/|//.*?$", "", line, flags=re.DOTALL)
            # check if line has \* using regex
            if re.search(r"/\*.*?", line):
                line = re.sub(r"/\*.*?$", "", line, flags=re.DOTALL)
                start_comment=True
                if line.strip():
                    function_lines.append(line)
                    # add to mapper
                    mapper[infunction_line]=infile_line
                    infunction_line+=1
            # check if line has *\ using regex
            if start_comment:
                if re.search(r".*\*/", line):
                    line = re.sub(r".*\*/", "", line, flags=re.DOTALL)
                    start_comment=False 
                # skip line if it is comment
                else:
                    continue
            # check if line not empty
            if line.strip():
                function_lines.append(line)
                # add to mapper
                mapper[infunction_line]=infile_line
                infunction_line+=1
    if in_function:
        functions.append({'function':'\n'.join(function_lines) , 'mapper':mapper})
    return functions

def predict_vul(model,tokenizer, args,function):
    """ given a function return if it is vulnerable or not and ranking of lines in the function according to their attention score
        which represent the participation of the line in the vulnerability
        example:
        input: function = "int main() {\n    printf("Hello World!");\n //comment \n   return 0;\n}"
        output: is_vul = 1
                ranking = [1,3,4,2]
    """       
    ucc_ds= UCC_Dataset(tokenizer, args,function=function)
    idx=0
    input_ids = ucc_ds.__getitem__(idx)['input_ids']
    attention_mask = ucc_ds.__getitem__(idx)['attention_mask']

    output,attentions = model(input_ids=input_ids.unsqueeze(dim=0), attention_mask=attention_mask.unsqueeze(dim=0),output_attentions=True)
    is_vul = torch.argmax(output).item()
    all_lines_score=[]
    ranking=None
    if is_vul:
        ids = input_ids.detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(ids)
        # this get one tensor that have attention for each token 
        attention=summerize_attention(attentions)    
        # clean att score for <s> and </s>
        attention=remove_special_token_score(attention) 
        # attention should be 1D tensor with seq length representing each token's attention value
        # size of both of them is 512
        assert len(all_tokens)==len(attention)
        word_att_scores = list(zip(all_tokens,attention)) #combine_lists(all_tokens, attention)
        all_lines_score,_=get_lines_score(word_att_scores)
        ranking = sorted(range(len(all_lines_score)), key=lambda i: all_lines_score[i], reverse=True)
        # if number of lines in function mor than 10 the n get top 10 lines expected the vulnerability to be in them
        if len(ranking)>10:
            ranking=ranking[:10]
    return is_vul,ranking
    
def Filevul(filename,model,tokenizer,args):
    """ given a file return list of lines that are vulnerable
        example:
        input: filename = "test.c"
        output: vul_lines_infile = [[1,3,4,2],[1,3,4,2]] # list of list of lines in each function that are vulnerable
    """
    code = read_c_file(filename)
    functions = tokenize_c_function(code)

    # print(functions[0])
    # print(predict_vul(model,tokenizer,args,functions[0]))
    vul_lines_infile=[]
    for function in functions:
        #print(function['function'])
        is_val,ranking=predict_vul(model,tokenizer,args,function['function'])
        #print(ranking)
        if is_val:
            vul_lines_infunc=[]
            mapper=function['mapper']
            #print(mapper)
            for line in ranking:
                # mapper map from function line level to file line level
                vul_lines_infunc.append(mapper[line+1])
            vul_lines_infile.append(vul_lines_infunc)        
    return vul_lines_infile


def Textvul(code,model,tokenizer,args):
    """ given a text return list of lines that are vulnerable
        example:
        input: code = "int main() {\n    printf("Hello World!");\n //comment \n   return 0;\n}"
        output: vul_lines_infile = [[1,3,4,2],[1,3,4,2]] # list of list of lines in each function that are vulnerable
    """
    functions = tokenize_c_function(code)
    #logging.info(f'number of functions in file {code} is {functions}')
    vul_lines_infile=[]
    for function in functions:
        #print(function['function'])
        is_val,ranking=predict_vul(model,tokenizer,args,function['function'])
        #print(ranking)
        if is_val:
            vul_lines_infunc=[]
            mapper=function['mapper']
            #print(mapper)
            for line in ranking:
                # mapper map from function line level to file line level
                vul_lines_infunc.append(mapper[line+1])
            vul_lines_infile.append(vul_lines_infunc)        
    return vul_lines_infile


def Directoryvul(dirname,model,tokenizer,args):
    """ given a directory return list of lines that are vulnerable in each file
        example:
        input: dirname = "test"
        output: vul_lines_indir = [("test.c",[[1,3,4,2],[1,3,4,2]]),("test2.c",[[1,3,4,2],[1,3,4,2]])] # list of list of lines in each function that are vulnerable
    """
    file_paths = get_c_and_cpp_files(dirname)
    vul_lines_indir=[]
    for file in file_paths:
        vul_lines_infile=Filevul(file,model,tokenizer,args)
        vul_lines_indir.append((file,vul_lines_infile))
    return vul_lines_indir


