# -*- coding:utf-8 -*-
import time
import xlrd
import pickle
import codecs
import re
import numpy as np
from urllib.parse import urlparse

def write_txt(data, path_detail):
    writer = codecs.open(path_detail, "w", encoding='utf-8', errors='ignore')
    for item in data:
        writer.write(item + '\n')
    writer.flush()
    writer.close()
#
# def calc_count(motion_result):
#     result_list = []
#     for line in motion_result.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         result_list.append(terms[0])
#     return result_list
#
def sort(motion_result):
    result_list = []
    for line in motion_result.readlines():
        line = line.strip()
        terms = line.split('\t')
        result_list.append(terms[0])
    return result_list

def user_aggragation(motion_result):
    DictQuery = {}
    DictTitle = {}
    count = 0
    for line in motion_result:
        line = line.strip()
        line = clean_str(line.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
        terms = line.split('\t')
        if len(terms) < 5:
            continue
        id = terms[0]
        query = terms[3]
        title = terms[4]
        if id not in DictQuery:
            DictQuery[id] = []
        if len(query) > 0:
            DictQuery[id].append(query)
        if id not in DictTitle:
            DictTitle[id] = []
        if len(title) > 0:
            DictTitle[id].append(title)
    return DictQuery, DictTitle

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\|", "", string)
    string = re.sub(r"\-", "", string)
    return string.strip().lower()

def write_dict(label, query, title, path_detail):
    writer = codecs.open(path_detail, "w", encoding='utf-8', errors='ignore')
    for each_key in query.keys():
        if(len(query[each_key]) + len(title[each_key]) > 10):
            writer.write(label+'\t'+"#".join(title[each_key])+'\t'+"#".join(query[each_key])+'\n')
    writer.flush()
    writer.close()
#
# def calc_count_eachuser(file):
#     result_dict = {}
#     for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         if terms[0] in result_dict:
#             result_dict[terms[0]] += int(terms[2])
#         result_dict[terms[0]] = int(terms[2])
#     return result_dict

# file = codecs.open('./data/jan_data/test/car.txt', "r", encoding='utf-8', errors='ignore')
# file1 = codecs.open('./data/car_eng_data/new/union.txt', "r", encoding='utf-8', errors='ignore')

# result_list = []
#
# for line in file.readlines():
#     line = line.strip()
#     result_list.append(line)
# np.random.shuffle(result_list)
# write_txt(result_list, './data/jan_data/car.txt')

# file1 = codecs.open('./data/pet_eng_data/pet_negative_test_data.txt', "r", encoding='utf-8', errors='ignore')
# url_dict, query_dict, title_dict = user_aggragation(file)
# write_dict(url_dict, query_dict, title_dict, "./data/car_eng_data/car_positive_test_data_current.txt")

# file = codecs.open('./data/car_eng_data/car_positive_test_data_previous.txt', "r", encoding='utf-8', errors='ignore')
# file1 = codecs.open('./data/car_eng_data/car_negative_test_data_previous.txt', "r", encoding='utf-8', errors='ignore')
# result_list = []
# for line in file.readlines():
#     line = line.strip()
#     terms = line.split('\t')
#     if (len(terms) < 3):
#         continue
#     query = terms[1]
#     title = terms[2]
#     query = clean_str(query.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
#     title = clean_str(title.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
#     text = title + "#" + query
#     sentences = text.split('#')
#     new_sentences = []
#     for sentence in sentences:
#         if len(sentence) > 3:
#             new_sentences.append(sentence)
#     sentences = new_sentences
#     if len(sentences) < 6:
#         continue
#     else:
#         result_list.append('1'+'\t'+query+'\t'+title)
# for line in file1.readlines():
#     line = line.strip()
#     terms = line.split('\t')
#     if (len(terms) < 3):
#         continue
#     query = terms[1]
#     title = terms[2]
#     query = clean_str(query.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
#     title = clean_str(title.encode('ascii', 'ignore').decode('utf-8', 'ignore'))
#     text = title + "#" + query
#     sentences = text.split('#')
#     new_sentences = []
#     for sentence in sentences:
#         if len(sentence) > 3:
#             new_sentences.append(sentence)
#     sentences = new_sentences
#     if len(sentences) < 6:
#         continue
#     else:
#         result_list.append('0' + '\t' + query + '\t' + title)
# write_txt(result_list, './data/car_eng_data/test_previous_car.tsv')



# count = calc_count(file)
# print(len(set(count)))

# file = codecs.open('./data/pet/pet_negative_user.txt', "r", encoding='utf-8', errors='ignore')
# file1 = codecs.open('./data/pet/click_test_user.txt', "r", encoding='utf-8', errors='ignore')
# result = sort(file)
# result1 = sort(file1)
# union_result = set(result) & set(result1)
# write_txt(set(result), "./data/pet/negative_user.txt")



# rank_dict = calc_count_eachuser(file)
# rank_dict = sorted(rank_dict.items(), key=lambda d: d[1], reverse=True)
# writer = codecs.open("./new_data/negative_user_rank2017-12-29_2017-12-31.txt", "w", encoding = 'utf-8', errors = 'ignore')
# for each_user in rank_dict:
#     writer.write(str(each_user[0])+'\n')
# writer.flush()
# writer.close()

#
# from urllib.parse import urlparse
# domain_list = []
# website_file = codecs.open('./data/pet_eng_data/pet_website.txt', "r", encoding='utf-8', errors='ignore')
# for line in website_file.readlines():
#     domain_list.append(line)
# print(domain_list)
# data_file = codecs.open('./data/pet_eng_data/negative_petuser_info_agg.txt', "r", encoding='utf-8', errors='ignore')
# data_list = []
# for line in data_file.readlines():
#     line = line.strip()
#     terms = line.split('\t')
#     if len(terms) > 1:
#         urls = terms[1].split('#')
#         for url in urls:
#             parsed_uri = urlparse(url)
#             domain = '{uri.netloc}'.format(uri=parsed_uri)
#             if domain+'\r\n' in domain_list:
#                 break
#         else:
#             data_list.append('0'+'\t'+terms[2]+'\t'+terms[3]+'\t')
# writer = codecs.open('./data/pet_eng_data/negative_user_test_data.txt', "w", encoding='utf-8', errors='ignore')
# for item in data_list:
#     writer.write(item + '\n')
# writer.flush()
# writer.close()
#


# file = codecs.open('./new_data/test/test_combine_new.tsv', "r", encoding='utf-8', errors='ignore')
# def calc_count_each_user(file):
#     result_list = []
#     for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         result_list.append(terms[1].split('#'))
#     return result_list
#
# res = calc_count_each_user(file)
# len_dict = {}
# for sent in res:
#     for i in range(len(sent)):
#         # print(len(sent[i]))
#         if len(sent[i]) not in len_dict:
#             len_dict[len(sent[i])] = 1
#         else:
#             len_dict[len(sent[i])] = len_dict[len(sent[i])] + 1
# # print(length)
# s = sorted(len_dict.keys())
# summary = 0
#
# length = sum(len_dict.values())
# print(length/len(res))
# for i in s:
#     summary = summary + len_dict[i]
#     if summary>length:
#         print(i)
#         break
#
# # print(sorted(len_dict.values(), reverse=True))
#
#
# __import__("ipdb").set_trace()
# print('sdfs')

# file = codecs.open('./data/pet_eng_data/new/pet_positive_user2018-01-15_2018-01-28.txt', "r", encoding='utf-8', errors='ignore')
# def split(file):
#     result_list1 = []
#     result_list2 = []
#     for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         if int(terms[1]) > 20180121:
#             result_list1.append(line.strip())
#         else:
#             result_list2.append(line.strip())
#     return result_list1, result_list2
# result_list1, result_list2 = split(file)
# write_txt(result_list1, "./data/pet_eng_data/new/pet_positive_user2018-01-22_2018-01-28.txt")
# write_txt(result_list2, "./data/pet_eng_data/new/pet_positive_user2018-01-15_2018-01-21.txt")

# web = codecs.open('./data/pre/pet_website.txt', "r", encoding='utf-8', errors='ignore')
# train_web_list = []
# test_web_list = []
# count = 0
# for line in web.readlines():
#     line = line.strip()
#     if count % 5 < 3:
#         train_web_list.append(line)
#     else:
#         test_web_list.append(line)
#     count += 1
# write_txt(train_web_list, "./data/pre/pet_website_train.txt")
# write_txt(test_web_list, "./data/pre/pet_website_test.txt")
#
# test_web = codecs.open('./data/pet_website_test.txt', "r", encoding='utf-8', errors='ignore')
# train_web = codecs.open('./data/pet_website_train.txt', "r", encoding='utf-8', errors='ignore')
# test_web_list = []
# for line in test_web.readlines():
#     line = line.strip()
#     test_web_list.append(line)
#
# train_web_list = []
# for line in train_web.readlines():
#     line = line.strip()
#     train_web_list.append(line)
#
# file = codecs.open('./data/pet/pet_negative_user_info2018-01-29_2018-02-18.txt', "r", encoding='utf-8', errors='ignore')
# result_train = []
# result_test = []
# dirty_negative = []
# for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         parsed_uri = urlparse(terms[2])
#         domain = '{uri.netloc}'.format(uri=parsed_uri)
#         if domain in train_web_list:
            # result_train.append(line)
        #     dirty_negative.append(terms[0])
        # elif domain in test_web_list:
            # result_test.append(line)
            # dirty_negative.append(terms[0])
        # else:
            # clean_negative.append(line)
# write_txt(result_train, "./data/pet/pet_positive_user_train.txt")
# write_txt(result_test, "./data/pet/pet_positive_user_test.txt")
# write_txt(dirty_negative, "./data/pet/pet_negative_user.txt")

# test_user = codecs.open('./data/pet/click_test_user.txt', "r", encoding='utf-8', errors='ignore')
# train_user = codecs.open('./data/pet/dirty_negative_user.txt', "r", encoding='utf-8', errors='ignore')
# test_user_list = []
# for line in test_user.readlines():
#     line = line.strip()
#     test_user_list.append(line)
#
# train_user_list = []
# for line in train_user.readlines():
#     line = line.strip()
#     train_user_list.append(line)
#
# file = codecs.open('./data/pet/pet_negative_user_info2018-01-29_2018-02-18.txt', "r", encoding='utf-8', errors='ignore')
# result_train = []
# result_test = []
# others = []
# for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         id = terms[0]
#         if id not in train_user_list:
#             result_train.append(line)
#         elif id in test_user_list:
#             result_test.append(line)
#         else:
#             continue
            # others.append(line)
# result_train.sort()
# result_test.sort()
# write_txt(result_train, "./data/pet/pet_negative_user_info2018-01-29_2018-02-18.txt")
# write_txt(result_test, "./data/pet/positive_click_test_info.txt")
# write_txt(others, "./data/pet/others.txt")
#
# web = codecs.open('./data/pet_website.txt', "r", encoding='utf-8', errors='ignore')
# web_list = []
# for line in web.readlines():
#     line = line.strip()
#     web_list.append(line)
#
# file = codecs.open('./data/pet/pet_negative_user_info2018-01-29_2018-02-18.txt', "r", encoding='utf-8', errors='ignore')
# combine_records = []
# useful_query = []
# current_records = []
# previous_records = []
# for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         if len(terms) < 5:
#             print(line)
        # parsed_uri = urlparse(terms[2])
        # domain = '{uri.netloc}'.format(uri=parsed_uri)
        # if domain in web_list:
        #     # useful_query.append(terms[3])
        #     continue
        # else:
        # combine_records.append(line)
            # if int(terms[1]) >= 20180122:
            #     current_records.append(line)
            # else:
            #     previous_records.append(line)
#
# new_combine_records = []
# for record in combine_records:
#     terms = record.split('\t')
#     if terms[3] in useful_query:
#         continue
#     else:
#         new_combine_records.append(record)

# combineDictQuery, combineDictTitle = user_aggragation(combine_records)
# write_dict('0', combineDictQuery, combineDictTitle, './data/pet/pet_negative.txt')
# currentDictQuery, currentDictTitle = user_aggragation(current_records)
# write_dict('1', currentDictQuery, currentDictTitle, './data/jan_data/pet_positive_testuser_info_previous.txt')
# previousDictQuery, previousDictTitle = user_aggragation(previous_records)
# write_dict('1', previousDictQuery, previousDictTitle, './data/jan_data/pet_positive_testuser_info_current.txt')

# file = codecs.open('./data/pre/pet_positive_user2018-01-29_2018-02-18.txt', "r", encoding='utf-8', errors='ignore')
# user_count = {}
# for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         if terms[0] not in user_count:
#             user_count[terms[0]] = 0
#         user_count[terms[0]] +=1
#
# user_count = sorted(user_count.items(), key=lambda d: d[1], reverse=True)
# writer = codecs.open('./data/pre/pet_positive_user_count1.txt', "w", encoding='utf-8', errors='ignore')
# for item in user_count:
#     writer.write(str(item[0]) + '\t' + str(item[1]) + '\n')
# writer.flush()
# writer.close()

# file = codecs.open('./data/pre/pet/pet_negative_user2018-01-29_2018-02-18.txt', "r", encoding='utf-8', errors='ignore')
# user_list = []
# for line in file.readlines():
#         line = line.strip()
#         terms = line.split('\t')
#         user_list.append(terms[0])
# write_txt(set(user_list), "./data/pre/pet/pet_negative_user.txt")
