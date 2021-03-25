import json


list_file = open("texts_list.txt", "r")
list_lines = list_file.readlines()

info_texts = open("info_texts.json", "r")
json_objects = json.load(info_texts)
info_texts.close()

outfile = open("new_info_texts.json", "w")

data = {}

for line in list_lines:

    id = line[:-1]

    data = {
        'id': id,
        'img_url': json_objects[id]['img_url'],
        'labels': json_objects[id]['labels'],
        'tweet_url': json_objects[id]['tweet_url'],
        'tweet_text': json_objects[id]['tweet_text'],
        'labels_str': json_objects[id]['labels_str'],
    }

    json.dump(data, outfile)
    outfile.write('\n')

outfile.close()
