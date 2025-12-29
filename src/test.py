l = [ None,
      None,
      None,
      44,
      44,
      44,
      None,
      None,
      None,
      None,
      None,
      None,
      44,
      None,
      44,
      None,
      44,
      44,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      44,
      44,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      44,
      44,
      None,
      None,
      None,
      44,
      44,
      44,
      44]

def pack(a_list):
    out = []
    count = 1
    prev = a_list[0]

    for item in a_list[1:]:
        if item != prev:
            out.append((prev, count))
            count = 1
            prev = item
        else:
            count += 1

    out.append((prev, count))
    return out

def unpack(a_list):
    out = []

    item_identifier = 0
    number_of_items = 1

    for item in a_list:
        out += [item[item_identifier]] * item[number_of_items]

    return out

def block_fill(a_list, error_threshold : int):

    item_identifier = 0
    number_of_items = 1

    packed_list = pack(a_list)
    for i, item in enumerate(packed_list[1:len(packed_list) - 1]): #correct error by filling in short None-s between data
        left_item_identifier = packed_list[i][item_identifier]
        right_item_identifier = packed_list[i + 2][item_identifier]

        if (left_item_identifier == right_item_identifier is not None) and item[number_of_items] <= error_threshold:
            packed_list[i + 1] = (left_item_identifier, item[1])

    for i, item in enumerate(packed_list[1:len(packed_list) - 1]): #correct error by erasing in short data between None-s
        left_item_identifier = packed_list[i][item_identifier]
        right_item_identifier = packed_list[i + 2][item_identifier]

        if (left_item_identifier == right_item_identifier is None) and item[number_of_items] <= error_threshold:
            packed_list[i + 1] = (left_item_identifier, item[1])

    return unpack(packed_list)


print(l)
print(block_fill(l ,3))
