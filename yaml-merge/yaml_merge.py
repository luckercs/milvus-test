import sys
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.tokens import CommentToken

def merge_yaml_files(large_file_path, small_file_path, output_file_path=None):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 10000
    yaml.default_flow_style = False
    yaml.sort_base_mapping_type_on_output = False
    yaml.allow_duplicate_keys = False

    with open(large_file_path, 'r', encoding='utf-8') as f:
        large_data = yaml.load(f)

    with open(small_file_path, 'r', encoding='utf-8') as f:
        small_data = yaml.load(f)

    def deep_merge(target, source):
        if isinstance(target, CommentedMap) and isinstance(source, CommentedMap):
            for key, value in source.items():
                if key in target:
                    key_pos = target.ca.items.get(key, None)

                    if isinstance(target[key], CommentedMap) and isinstance(value, CommentedMap):
                        deep_merge(target[key], value)
                    elif isinstance(target[key], CommentedSeq) and isinstance(value, CommentedSeq):
                        target[key].extend([item for item in value if item not in target[key]])
                    else:
                        target[key] = value

                    if key_pos is not None:
                        target.ca.items[key] = key_pos
                else:
                    target[key] = value
        return target

    merged_data = deep_merge(large_data, small_data)

    if output_file_path is None:
        output_file_path = large_file_path

    with open(output_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(merged_data, f)

    print(f"YAML merge success，stored as : {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python merge_yaml_precise.py target.yaml patch.yaml result.yaml")
        sys.exit(1)

    large_file = sys.argv[1]
    small_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    merge_yaml_files(large_file, small_file, output_file)
