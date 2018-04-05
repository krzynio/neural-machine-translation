import re

import unidecode


def remove_repeating_spaces(sentence):
    return re.sub('[ ]{2,}', ' ', sentence)


def remove_non_alphanumeric(sentence):
    return re.sub('[^0-9a-zA-Z ]+', '', sentence)


def remove_diacritics(sentence):
    return unidecode.unidecode(sentence)


def tokenize(sentence):
    return sentence.split(' ')


def prepare_sentence(sentence):
    prepared = remove_repeating_spaces(remove_non_alphanumeric(remove_diacritics(sentence))).strip()
    tokenized = tokenize(sentence)
    return prepared, len(tokenized)


def create_prepared_data(src_sentences_file,
                         dst_sentences_file,
                         prepared_src_file,
                         prepared_dst_file,
                         validation_src_file,
                         validation_dst_file,
                         size=None,
                         validation_size=0,
                         min_length=1,
                         max_length=40):
    with open(src_sentences_file) as source_file:
        with open(dst_sentences_file) as dst_file:
            with open(prepared_src_file, 'w') as output_src_file:
                with open(prepared_dst_file, 'w') as output_dst_file:
                    with open(validation_src_file, 'w') as output_val_src_file:
                        with open(validation_dst_file, 'w') as output_val_dst_file:
                            val_total = 0
                            total = 0
                            while size is None or total < size:
                                try:
                                    source_sentence = next(source_file)
                                    dst_sentence = next(dst_file)
                                except:
                                    return

                                source_sentence, src_len = prepare_sentence(source_sentence)
                                dst_sentence, dst_len = prepare_sentence(dst_sentence)
                                if src_len > max_length or src_len < min_length or dst_len > max_length or dst_len < min_length:
                                    continue

                                if val_total < validation_size:
                                    val_total += 1
                                    output_val_src_file.write(source_sentence + '\n')
                                    output_val_dst_file.write(dst_sentence + '\n')
                                else:
                                    total += 1
                                    output_src_file.write(source_sentence + '\n')
                                    output_dst_file.write(dst_sentence + '\n')
