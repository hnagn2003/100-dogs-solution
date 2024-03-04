import json
import jsonlines

def add_new_feature(coco_json_path, output_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Iterate through annotations and add the new feature
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        new_feature = f"label_{category_id}_bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        annotation['custom_annotation'] = new_feature

    # Write the modified data to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
import json
import jsonlines

def convert_coco_to_custom_format(input_file, output_file):
    with open(input_file, 'r') as infile, jsonlines.open(output_file, 'w') as outfile:
        coco_data = json.load(infile)

        for image_info in coco_data["images"]:
            file_name = image_info["file_name"]
            image_id = image_info["id"]

            objects_info = {"custom_annotation": []}
            for annotation in coco_data["annotations"]:
                if annotation["image_id"] == image_id:
                    custom_annotation = annotation["custom_annotation"]
                    objects_info["custom_annotation"].append(custom_annotation)

            custom_format_entry = {"file_name": file_name, "objects": objects_info}
            outfile.write(custom_format_entry)

# Replace 'coco.json' with the name of your input JSON file and 'custom_format.jsonl' with the desired output JSON Lines file.


# Example usage
input_json_file = './tiny-coco/small_coco/metadata.json'
output_json_file = './tiny-coco/small_coco/train/metadata2.jsonl'

convert_coco_to_custom_format(input_json_file, output_json_file)
