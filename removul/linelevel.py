from __future__ import absolute_import, division, print_function
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)

# get flow lines as array of lines
def get_all_flaw_lines(flaw_lines: str, flaw_line_seperator: str) -> list: 
    """ 
        get all flaw lines as list of strings
        flaw_lines: str -> string of flaw lines seperated by flaw_line_seperator
        example:
        flaw lines: "line1/~/line2/~/line3"
        flaw_line_seperator: "/~/"
        return: ["line1", "line2", "line3"]
    """
    if isinstance(flaw_lines, str):
        # strip -> remove the leading and trailing  flaw_line_seperator in the string
        flaw_lines = flaw_lines.strip(flaw_line_seperator)
        # split on the flaw line seperator
        flaw_lines = flaw_lines.split(flaw_line_seperator)
        # remove any space in the begane or end of the line
        flaw_lines = [line.strip() for line in flaw_lines]
    else:
        flaw_lines = []
    return flaw_lines

def encode_one_line(line: str, tokenizer): 
    """ encode one line of code to tokens 
        line: str -> string of code line
        tokenizer: tokenizer object
        example:
        line: "int x=15"
        return: ["int", "x", "=", "15"]
        add "@ " at the beginning to ensure the encoding consistency, i.e., previous -> previous, not previous > pre + vious
        what is tokenizer.tokenize() doing? -> https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.tokenize
        The "@" symbol is added to the beginning of the input string to indicate the start of a new line or code block.
        This is because some tokenizers, may use the context of the previous lines of code to tokenize the current line correctly.
        Adding the "@" symbol ensures that the tokenizer treats the input line as the start of a new block of code, without considering any previous lines of code.
        However, since the "@" symbol is not a part of the actual code, it is removed from the list of tokens in the final step of the function.
        The special character "Ġ" represents the start of a new word in some tokenizers
    """
    
    code_tokens = tokenizer.tokenize("@ " + line)
    return [token.replace("Ġ", "") for token in code_tokens if token != "@"]

def encode_all_lines(all_lines: list, tokenizer) -> list:
    """ encode all lines of code to tokens
        all_lines: list -> list of strings of code lines
        tokenizer: tokenizer object
        example:
        all_lines: ["int x=15", "int y=20"]
        return: [["int", "x", "=", "15"], ["int", "y", "=", "20"]]
        """
    encoded = []
    for line in all_lines:
        encoded.append(encode_one_line(line=line, tokenizer=tokenizer))
    # list of lists of tokens [[one,list,for,each,line],[]]    
    return encoded

def clean_special_token_values(all_values): 
    """ remove the attention score for special tokens <s> and </s>
        all_values: list -> list of attention scores for each token
        example:
        all_values: [0.1, 0.2, 0.3, 0.4, 0.5]
        return: [0, 0.2, 0.3, 0.4, 0]
    """
    # special token in the beginning of the seq 
    all_values[0] = 0
    # get the last non-zero value which represents the att score for </s> token
    idx = [index for index, item in enumerate(all_values) if item != 0][-1]
    all_values[idx] = 0
    return all_values


def get_all_lines_score(word_att_scores: list, verified_flaw_lines: list=[]):
    """ get the attention score for each line 
        and index of flaw lines
        word_att_scores: list -> list of lists [['word',score],...]
        verified_flaw_lines: list -> list of lists for the line tokens [['token','in', 'flaw','line'],[]]
        example:
        word_att_scores: [['int', 0.1], ['x', 0.2], ['=', 0.3], ['15', 0.4],['/~/',0.2], ['int', 0.5], ['y', 0.6], ['=', 0.7], ['20', 0.8]]
        verified_flaw_lines: [['int', 'x', '=', '15']]
        return: [0.1+0.2+0.3+0.4, 0.5+0.6+0.7+0.8], [0]
    """
    # get one list of strings each string represent flaw line
    verified_flaw_lines = [''.join(l) for l in verified_flaw_lines]
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    separator = ["Ċ", " Ċ", "ĊĊ"," ĊĊ", "ĊĊĊ"," ĊĊĊ"]
    # to return
    all_lines_score = []
    score_sum = 0
    line_idx = 0
    flaw_line_indices = []
    line = ""
    for i in range(len(word_att_scores)):
        # summerize if meet line separator or the last token
        word,score=word_att_scores[i]
        if ((word in separator) or (i == (len(word_att_scores) - 1))) and score_sum != 0:
            score_sum += score
            all_lines_score.append(score_sum)
            # check if the line in the flaw line list
            if line in verified_flaw_lines:
                flaw_line_indices.append(line_idx)
            line = ""
            score_sum = 0
            line_idx += 1
        # else accumulate score
        elif word not in separator:
            line += word
            score_sum += score
    # all_lines_score list of scores one score for each line
    # index of flaw lines (can use this index to access all_lines_score and get score for flaw line)
    return all_lines_score, flaw_line_indices

def line_level_evaluation(all_lines_score: list, flaw_line_indices: list):
    """ get number of correctly predicted flaw lines and min clean lines inspected which is IFA
        all_lines_score: list -> list of scores one score for each line
        flaw_line_indices: list -> index of flaw lines (can use this index to access all_lines_score and get score for flaw line)
        example:
        all_lines_score: [0.1+0.2+0.3+0.4, 0.5+0.6+0.7+0.8]
        flaw_line_indices: [0]
        return: 1, 0
    """
    # line indices ranking based on attr values get the indexes of higher score to lower score
    ranking = sorted(range(len(all_lines_score)), key=lambda i: all_lines_score[i], reverse=True)
    correctly_predicted_flaw_lines = 0
    all_clean_lines_inspected=[]
    # if within top-k
    # I think it should get min(len(all_lines_score),top_k) instead of len(all_lines_score) * top_k
    k = min(len(all_lines_score),10)
    for indice in flaw_line_indices:
        # if detecting flow line in any of top k
        if indice in ranking[: k]:
            #use to get top-10
            correctly_predicted_flaw_lines += 1
        
        # calculate Initial False Alarm
        # IFA counts how many clean lines are inspected until the first vulnerable line is found when inspecting the lines ranked by the approaches.
        flaw_line_index_in_ranking = ranking.index(indice)
        # e.g. flaw_line_idx_in_ranking = 3 will include 1 vulnerable line and 3 clean lines
        all_clean_lines_inspected.append(flaw_line_index_in_ranking)  
    # for IFA
    min_clean_lines_inspected = min(all_clean_lines_inspected)
    # append result for one top-k value
    return correctly_predicted_flaw_lines,min_clean_lines_inspected

# verifiy that flaw line is in the function after tokenization both flaw line and function
# flow_tokens is list of lists of tokens list of tokens for each flow line 
def verify_flaw_line_in_func( func_tokens: list, flow_tokens: list):
    """ verify that flaw line is in the function after tokenization both flaw line and function
    that is to check if the flaw line tokens is in the function tokens
        func_tokens: list -> list of tokens for the function
        flow_tokens: list -> list of lists of tokens list of tokens for each flow line 
        example:
        func_tokens: ["int", "x", "=", "15", "int", "y", "=", "20"]
        flow_tokens: [["int", "x", "=", "15"]]
        return: True, [["int", "x", "=", "15"]]
    """
    verify_flaw_line = []
    verified_one_line=False
    func_tokens_str = ''.join(func_tokens)
    for flow_token in flow_tokens:
        flow_token_str = ''.join(flow_token)
        if flow_token_str in func_tokens_str:
            verify_flaw_line.append(flow_token)
            verified_one_line=True
    return verified_one_line, verify_flaw_line 


# get the attention score for each 512 token
def summerize_attention(attentions):
    """ get the attention score for each 512 token
        attentions: list -> list of attentions for each layer dimintion: [1, 12, 512, 512]
        get one tensor that have attention for each token
    """
    #print(attentions[0].shape) # torch.Size([1, 12, 512, 512])
    attentions = attentions[0][0]
    #print(attentions.shape) # torch.Size([12, 512, 512])
    attention = None
    # go into each the layer for 12 layer
    for i in range(len(attentions)):
        layer_attention = attentions[i]
        # summerize the values of each token dot other tokens (this get one tensor that have value for each token )
        layer_attention = sum(layer_attention)
        # print(layer_attention.shape) # layer_attention
        if attention is None:
            attention = layer_attention
        else:
            attention += layer_attention
    return attention   


def LineLevel(dataloader, test_df, tokenizer, model):
    """ Line level evaluation"""
    progress_bar = tqdm(dataloader, total=len(dataloader))
    # loop over all test functions oe by one
    index=0
    line_sperator="/~/"
    Top_10_accuracy=0
    total_min_clean_lines_inspected=0
    total_functions=0
    for mini_batch in progress_bar:
        input_ids=mini_batch['input_ids']
        attention_mask=mini_batch['attention_mask']
        labels=mini_batch['labels']
        ids = input_ids[0].detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(ids)
        # remove the word of word
        # for each word it has Ġ to detect its begining
        all_tokens = [token.replace("Ġ", "") for token in all_tokens]
        all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
        
        if test_df.iloc[index]["flaw_line"] is not np.nan:
            # get list of flaw lines
            flow_lines_list=get_all_flaw_lines(test_df.iloc[index]["flaw_line"],line_sperator)
            # encode each flaw line
            flaw_tokens_encoded = encode_all_lines(all_lines=flow_lines_list, tokenizer=tokenizer)
            # verified_one_line is boolean get true if at least one flaw line is verified to be in the function
            # verified_flaw_lines is as flaw_tokens_encoded is list of lists of token each list repesent line
            verified_one_line, verified_flaw_lines=verify_flaw_line_in_func(all_tokens,flaw_tokens_encoded)
            # if at least one flow line found in the function
            if verified_one_line:
                _,_, attentions = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels, output_attentions=True)
                # this get one tensor that have attention for each token 
                attention=summerize_attention(attentions)    
                # clean att score for <s> and </s>
                attention=clean_special_token_values(attention) 
                # attention should be 1D tensor with seq length representing each token's attention value
                # size of both of them is 512
                assert len(all_tokens)==len(attention)
                word_att_scores = list(zip(all_tokens,attention)) #combine_lists(all_tokens, attention)
                # word_att_scores list of lists [['word',score],...] and verified_flaw_lines list of lists for the
                # line tokens [['token','in', 'flaw','line'],[]]
                all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines)
                # return if no flaw lines exist
                if len(flaw_line_indices) == 0:
                    logging.error(f'not verified{index}') 
                else:
                    correctly_predicted_flaw_lines,min_clean_lines_inspected = line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices)
                    if correctly_predicted_flaw_lines>0:
                        Top_10_accuracy+=1
                    total_min_clean_lines_inspected+=min_clean_lines_inspected   
                    total_functions+=1
            else:
                # for vulnarbility lines exceed 512 token
                logging.warning(f'not verified{index,labels.item()}')    
        index+=1
        
    logging.info(f'Top_10_Accuracy: {Top_10_accuracy/total_functions}')
    logging.info(f'IFA: {total_min_clean_lines_inspected/total_functions}')


        
