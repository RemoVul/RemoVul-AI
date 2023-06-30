import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)


def remove_special_token_score(scores: list): 
    """ remove the attention score for special tokens <s> and </s>
        scores : list -> list of attention scores for each token
        example:
        scores : [0.1, 0.2, 0.3, 0.4, 0.5]
        return: [0, 0.2, 0.3, 0.4, 0]
    """
    # special token in the beginning of the seq 
    scores[0] = 0
    # get the last non-zero value which represents the att score for </s> token
    index = [index for index, score in enumerate(scores) if score != 0][-1]
    scores[index] = 0
    return scores


def get_lines_score(word_score_list, verified_flaw_lines=[],separator=["Ċ", " Ċ", "ĊĊ"," ĊĊ", "ĊĊĊ"," ĊĊĊ","ĉ"]):
    """ get the attention score for each line 
        and index of flaw lines
        word_score_list: list -> list of tuble [('word',score),...]
        verified_flaw_lines: list -> list of lists for the line tokens [['token','in', 'flaw','line'],[]]
        example:
        word_scores: [['int', 0.1], ['x', 0.2], ['=', 0.3], ['15', 0.4],['/~/',0.2], ['int', 0.5], ['y', 0.6], ['=', 0.7], ['20', 0.8]]
        verified_flaw_lines: [['int', 'x', '=', '15']]
        return: [0.1+0.2+0.3+0.4, 0.5+0.6+0.7+0.8], [0]
    """
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    # to return
    score_for_each_line = []
    score_sum = 0
    line_index = 0
    index_of_flaw_lines = []
    line = ""
    for index, word_score in enumerate(word_score_list):
        word, score = word_score
        if word in separator or index == len(word_score_list)-1:
            if score_sum != 0:
                score_sum += score
                score_for_each_line.append(score_sum)
                score_sum = 0
                if line in verified_flaw_lines:
                    index_of_flaw_lines.append(line_index)
                line = ""
                line_index += 1
        elif word not in separator:
            score_sum += score
            line += word  
    # all_lines_score list of scores one score for each line
    # index of flaw lines (can use this index to access all_lines_score and get score for flaw line)
    return score_for_each_line, index_of_flaw_lines      

def get_evaluation_matrix(score_for_each_line, index_of_flaw_lines):

    """ get number of correctly predicted flaw lines and min clean lines inspected which is IFA
        score_for_each_line: list -> list of scores one score for each line
        index_of_flaw_lines: list -> index of flaw lines (can use this index to access all_lines_score and get score for flaw line)
        example:
        all_lines_score: [0.1+0.2+0.3+0.4, 0.5+0.6+0.7+0.8]
        flaw_line_indices: [0]
        return: 1, 0
    """
    # convert score_for_each_line to numpy array
    score_for_each_line = np.array([tensor.item() for tensor in score_for_each_line])
    # get ordered index of lines based on score
    ordered_index = np.argsort(score_for_each_line)[::-1]
    # get top k lines
    top_k_lines = ordered_index[:min(len(score_for_each_line), 10)]
    number_of_predicted_flaw_lines = 0
    clean_lines_before_get_firstvul = 0
    # number of predicted flaw lines in top k lines
    for line_index in top_k_lines:
        if line_index in index_of_flaw_lines:
            number_of_predicted_flaw_lines += 1
        elif number_of_predicted_flaw_lines == 0:
            clean_lines_before_get_firstvul += 1    

    return number_of_predicted_flaw_lines,clean_lines_before_get_firstvul 


# verifiy that flaw line is in the function after tokenization both flaw line and function
# flow_tokens is list of lists of tokens list of tokens for each flow line 
def verify_flaw_line_in_func( func_tokens: list, flow_tokens: list):
    """ verify that flaw line is in the function after tokenization both flaw line and function
    that is to check if the flaw line tokens is in the function tokens
        func_tokens: list -> list of tokens for the function
        flow_tokens: list -> list of lists of tokens list of tokens for each flow line 
        example:
        func_tokens: ["int", "x", "=", "15", "int", "y", "=", "20"]
        flow_tokens: [["int", "x", "=", "15"], ["int", "y", "=", "50"]]
        return: True, [["int", "x", "=", "15"]]
    """
    func_tokens_str = ''.join(func_tokens)
    verified_flaw_line =[ line_tokens for line_tokens in flow_tokens if ''.join(line_tokens) in func_tokens_str]
    return len(verified_flaw_line)>0 ,verified_flaw_line


# get the attention score for each 512 token
def summerize_attention(attentions):
    """ get the attention score for each 512 token
        attentions: list -> list of attentions for each layer dimintion: [1, 12, 512, 512]
        get one tensor that have attention for each token
    """
    #print(attentions[0].shape) # torch.Size([1, 12, 512, 512])
    attentions = attentions[0][0]
    #print(attentions.shape) # torch.Size([12, 512, 512])
    # sum the attention for each token in each layer
    attention = sum(sum(attentions[i]) for i in range(len(attentions)))
    #print(attention.shape) # torch.Size([512])      
    return attention   

def LineLevel(dataloader, test_df, tokenizer, model):
    """ Line level evaluation"""
    progress_bar = tqdm(dataloader, total=len(dataloader))
    # loop over all test functions oe by one
    index=0
    line_sperator="/~/"
    Top_10_accuracy=0
    total_clean_lines_before_get_firstvul=0
    total_functions=0
    # each mini_batch is a one function
    for mini_batch in progress_bar:
        input_ids=mini_batch['input_ids']
        attention_mask=mini_batch['attention_mask']
        labels=mini_batch['labels']
        ids = input_ids[0].detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(ids)
        #logging.info(f'tokenized function: {all_tokens}')
        # replace Ġ which represent the start of subword ex Ġof
        all_tokens = [w.replace('Ġ', '') for w in all_tokens]
        #  'Ċ', 'ĉ' are used to represent the start new line
        #logging.info(f'tokenized function: {all_tokens}')
        if test_df.iloc[index]["flaw_line"] is not np.nan:
            # get list of flaw lines   flaw lines: "line1/~/line2/~/line3" -> ["line1","line2","line3"]
            flaw_lines = test_df.iloc[index]["flaw_line"].split(line_sperator)
            # remove leading and trailing spaces in each line
            flaw_lines = [line.strip() for line in flaw_lines]
            # remove lines that is empty string
            flaw_lines = [line for line in flaw_lines if line]
            ######
            # add "@ " at the beginning to ensure the encoding consistency, i.e., previous -> previous, not previous > pre + vious
            # what is tokenizer.tokenize() doing? -> https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.tokenize
            # The "@" symbol is added to the beginning of the input string to indicate the start of a new line or code block.
            # This is because some tokenizers, may use the context of the previous lines of code to tokenize the current line correctly.
            # Adding the "@" symbol ensures that the tokenizer treats the input line as the start of a new block of code, without considering any previous lines of code.
            # However, since the "@" symbol is not a part of the actual code, it is removed from the list of tokens in the final step of the function.
            # The special character "Ġ" represents the start of a new word in some tokenizers
            # tokenize each line in flaw lines
            # ex: flaw lines: ["line1","line2","line3"] -> flaw lines tokens: [["line,"1"],["line,"2"],["line,"3"]]
            flaw_lines_tokens = [tokenizer.tokenize("@ " +line) for line in flaw_lines]
            # remove @ token and remove Ġ which represent the start of subword ex Ġof
            flaw_lines_tokens = [[w.replace('Ġ', '') for w in line if w !="@"] for line in flaw_lines_tokens]
            # verified_one_line is boolean get true if at least one flaw line is verified to be in the function
            # verified_flaw_lines is as flaw_tokens_encoded is list of lists of token each list repesent line
            verified_one_line, verified_flaw_lines=verify_flaw_line_in_func(all_tokens,flaw_lines_tokens)
            if verified_one_line:
                _,_, attentions = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels, output_attentions=True)
                # this get one tensor that have attention for each token 
                attention=summerize_attention(attentions)   
                # clean att score for <s> and </s>
                attention=remove_special_token_score(attention) 
                # attention should be 1D tensor with seq length representing each token's attention value
                # size of both of them is 512
                assert len(all_tokens)==len(attention)
                word_score_list = list(zip(all_tokens,attention)) #combine_lists(all_tokens, attention)
                #logging.info(f'word_score_list: {word_score_list}')
                #get one list of strings each string represent flaw line
                verified_flaw_lines = [''.join(l) for l in verified_flaw_lines]
                # word_att_scores list of lists [['word',score],...] and verified_flaw_lines list of lists for the
                # line tokens [['token','in', 'flaw','line'],[]]
                score_for_each_line, index_of_flaw_lines = get_lines_score(word_score_list, verified_flaw_lines)
                # return if no flaw lines exist
                if len(index_of_flaw_lines) == 0:
                    logging.error(f'not verified{index}')        
                else:
                    number_of_predicted_flaw_lines,clean_lines_before_get_firstvul = get_evaluation_matrix(score_for_each_line=score_for_each_line, index_of_flaw_lines=index_of_flaw_lines)
                    if number_of_predicted_flaw_lines>0:
                        Top_10_accuracy+=1
                    total_clean_lines_before_get_firstvul+=clean_lines_before_get_firstvul   
                    total_functions+=1
            else:
                # for vulnarbility lines exceed 512 token
                logging.warning(f'not verified{index,labels.item()}')    
        index+=1
    logging.info(f'Top_10_Accuracy: {Top_10_accuracy/total_functions}')
    logging.info(f'IFA: {total_clean_lines_before_get_firstvul/total_functions}')

