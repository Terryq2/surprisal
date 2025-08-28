import os
from pprint import pprint
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Driver:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side = "left")
        self.tokenizer.pad_token = self.tokenizer.eos_token


        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, encoding='utf-8') as json_data:
            self.config = json.load(json_data)

        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def conjoin_words(self, word_list: list[str]) -> str:
        """
        Conjoins a list of words into a full string with spaces between consecutive words
        """
        conjoined: str = ""
        for i in range(0, len(word_list)-1):
            conjoined += word_list[i] + " "
        conjoined += word_list[len(word_list)-1]
        return conjoined

    def to_tokens_and_logprobs(self,
                               input_texts):
        '''
        Outputs the log probs of the input texts
        '''
        padded = [self.tokenizer.eos_token + self.tokenizer.eos_token + text for text in input_texts]

        input_ids = self.tokenizer(padded,
                            padding=True,
                            return_tensors="pt").input_ids
        outputs = self.model(input_ids)

        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        batch = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token not in self.tokenizer.all_special_ids:
                    text_sequence.append((self.tokenizer.decode(token), p.item()))
            batch.append(text_sequence)
        return batch
    
    def get_all_sentences(self,
                          path: str):
        """
        Gets every single sentence in the input csv
        """
        df: pd.DataFrame = pd.read_csv(path)

        all_lists = []
        running_list = []
        current_sentence = 1
        for _, row in df.iterrows():
            sentence = row['Passage']
            if sentence != current_sentence:
                all_lists.append(running_list)
                running_list = []
                current_sentence = sentence

            current_word = row['WordWithPunctuation']

            running_list.append(current_word)

        all_lists.append(running_list)

        output_raw = []
        for sentence in all_lists:
            output_raw.append(self.conjoin_words(sentence))

        return output_raw
    
    def _add_up(self,
                prelim_results,
                sentences_raw):
        """
        Adds up the OOV vocabs surprisal 
        """
        result = []
        for index, raw in enumerate(sentences_raw):
            word_ptr = 0
            sentence = prelim_results[index]
            words_to_match = raw.split()
            composed_word = ""
            composed_word_log_prob = 0
            for partial_words, log_prob in sentence:
                composed_word += partial_words.strip()
                composed_word_log_prob += -log_prob
                if composed_word == words_to_match[word_ptr]:
                    result.append(composed_word_log_prob)
                    word_ptr += 1
                    composed_word = ""
                    composed_word_log_prob = 0
        return result

    def write_surprisal(self):
        """
        Writes the surprisal of the words of the input file into a seperate column
        """
        os.makedirs("output_files", exist_ok = True)

        for name in self.config['INPUT_FILE_NAMES']:
            
            
            real_path = os.path.join("input_files", f"{name}.csv")
            sentences_raw = self.get_all_sentences(real_path)
            prelim_results = self.to_tokens_and_logprobs(sentences_raw)

            final_result = self._add_up(prelim_results, sentences_raw)

            df: pd.DataFrame = pd.read_csv(real_path)

            df['surprisal'] = final_result
            df.to_csv(f'output_files/{name}_processed.csv', index=False)


if __name__ == "__main__":
    driver = Driver()
    driver.write_surprisal()
    
    
    


            
            
            

