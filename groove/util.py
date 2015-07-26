# groove - a graph tool for vector embeddings
# Hugo Gascon <hgascon@mail.de>


def ngrams(tokens, min_n=2, max_n=2):
    """
    Find all ngrams within a series of tokens

    :param tokens: a list of strings
    """
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+min_n, min(n_tokens, i+max_n)+1):
            yield tokens[i:j]

def get_ngrams_from_str_list(tokens, delimiter='', post=True):
    if delimiter:
        i = 0
        if post: i = 1
        features = list(ngrams(''.join([t.split(delimiter)[i]
                               for t in tokens])))
    else:
        features = list(ngrams(' '.join(tokens)))

    for i,f in enumerate(features):
        try:
            features[i] = f.decode('utf-8')
        except UnicodeDecodeError:
            print "Unknown encoding found! ({}) " \
                  "[character ignored]".format(f)
            features[i] = unicode(f, errors='ignore')
        except UnicodeEncodeError:
            print "UnicodeEncodeError: {}".format(f)
            features[i] = f

    return features

