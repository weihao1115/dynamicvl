from prettytable import PrettyTable, HRuleStyle


def dict_to_table(data_dict, headers=None):
    table = PrettyTable()
    table.hrules = HRuleStyle.ALL

    max_list_length = max(len(value) for value in data_dict.values())

    if headers is None:
        headers = ["名称"] + [f"值{i + 1}" for i in range(max_list_length)]

    table.field_names = headers

    for key, value_list in data_dict.items():
        row = [key] + value_list + [""] * (max_list_length - len(value_list))
        table.add_row(row)

    return table
