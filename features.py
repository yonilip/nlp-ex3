def parse_rule(rule):
    which_word = rule.split("=")[0]
    which_tag = rule.split("_")[1]
    word = rule.split("=")[1].split("_")[0]
    if which_word == "word-1":
        index_diff = -1
    elif which_word in ["curword", "prefix", "suffix"]:
        index_diff = 0
    else:
        index_diff = 1
    return which_word, index_diff, word, which_tag


def make_feature_function(features_txt):
    """
    make a feature function using the specified features
    in the given text
    """
    rules_by_tag = {}
    for rule_i,line in enumerate(features_txt.splitlines()):
        which_word, index_diff, word, which_tag = parse_rule(line)
        if which_tag not in rules_by_tag:
            rules_by_tag[which_tag] = []
        rules_by_tag[which_tag].append((which_word, index_diff, word, which_tag, rule_i))

    def f(words, prev_tag, tag, index):
        result = [0] * 100
        if tag not in rules_by_tag:
            return result
        for rule in rules_by_tag[tag]:

            which_word, index_diff, word, which_tag, rule_i = rule
            word_index = index + index_diff

            ### check whether this rule applies ###

            # check that index is ok
            if word_index < 0 or word_index >= len(words):
                continue

            # check the word condition
            if which_word in ["prefix", "suffix"]:
                if which_word == "prefix":
                    if not words[index].startswith(word):
                        continue
                if which_word == "suffix":
                    if not words[index].endswith(word):
                        continue
            elif words[word_index] != word:
                continue

            # word/prefix/suffix is ok, just check the tag
            if tag == which_tag:
                result[rule_i] = 1
        return result
    return f


