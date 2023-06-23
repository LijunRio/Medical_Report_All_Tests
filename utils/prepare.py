import json

def annotation2captions(annotation_path, captions_path):

    captions = {}

    with open(annotation_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    data = data['train'] + data['val'] + data['test']

    for item in data:
        captions[item["image_path"][0]] = item["finding"]
    
    with open(captions_path, 'w', encoding='utf-8-sig') as f:
        json.dump(captions, f, ensure_ascii=False)

def generate_file_list(annotation_path, file_list_path, split):

    file_list_path = file_list_path + split + '_data.txt'
    ids = []

    with open(annotation_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)

    data = data[split]

    for item in data:
        id = str(item['uid']) + '_1\n'
        ids.append(id)

    with open(file_list_path, 'w', encoding='utf-8-sig') as f:
        f.writelines(ids)

if __name__ == '__main__':
    annotation2captions('../dataset/Ultrasonic_datasets/Throid_dataset/new_Thyroid2.json', '../data/new_data/CN/Thyroid/captions.json')
    generate_file_list('../dataset/Ultrasonic_datasets/Throid_dataset/new_Thyroid2.json' , '../data/new_data/CN/Thyroid/', 'train')
    generate_file_list('../dataset/Ultrasonic_datasets/Throid_dataset/new_Thyroid2.json' , '../data/new_data/CN/Thyroid/', 'val')
    generate_file_list('../dataset/Ultrasonic_datasets/Throid_dataset/new_Thyroid2.json' , '../data/new_data/CN/Thyroid/', 'test')
