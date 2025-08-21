from power.levenshtein import Levenshtein, ExpandedAlignment
from power.aligner import PowerAligner
import unittest
import pdb
pdb.set_trace()

# chinese word phonemization
from phonemizer import phonemize
from phonemizer.separator import Separator

def chinese_word_phonemization(text):
    # text = [
    #         "三不孜儿里",
    #         "散不走的"
    #         ]#[line.strip() for line in text.split('\n') if line]

    separator = Separator(phone=' ', word=' || ')
    # phn is a list of 190 phonemized sentences
    phn = phonemize(
        text,
        language='cmn',
        backend='espeak',
        # separator=Separator(phone=None, word=' ', syllable='|'),
        strip=True,
        preserve_punctuation=True,
        separator=separator,
        njobs=4)
    
    return phn






def test_lev_phones1(refwords, hypwords):
    # ref =   [ "",  "",  "",   "",   "|", "#", "b", "uh", "ch", "#", "",   "er","#", "ih", "ng", "|" ]
    # hyp =   [ "|", "#", "dh", "ax", "|", "#", "m", "ax", "ch", "#", "uh", "r", "#", "ih", "ng", "|" ]
    # align = [ 'I', 'I', 'I',  'I',  'C', 'C', 'S', 'S',  'C',  'C', 'I',  'S', 'C', 'C',  'C',  'C' ]
    # refwords = [x for x in ref if x]
    # hypwords = [x for x in hyp if x]
    weights = {'C': 0, 'S': 4, 'D': 3, 'I': 3}
    reserve_list = {'||', '#'}
    exclusive_sets = \
    [{'ao', 'ah', 'ow', 'ae', 'er', 'ih', 'eh', 'aa', 'uw', 'ay', 'ax', 'uh', 'aw', 'axr', 'oy', 'ey', 'iy'}, 
    {'ts', 'x', 'kh', 'tɕ', 'tsh', 't', 'dx', 'z', 'k', 'm', 'zh', 'y', 'b', 'f', 'eng', 'dh', 'p', 'l', 'r', 'ch', 'em', 'q', 'el', 'd', 'jh', 'sh', 'ng', 'w', 'en', 'g', 'nx', 's', 'hh', 'th', 'v', 'n'}, 
    {'r', 'axr', 'er'}]
    lev = Levenshtein.align(refwords, hypwords, lowercase=True,
                            weights=weights,
                            reserve_list=reserve_list, 
                            exclusive_sets=exclusive_sets
                        )
    lev.editops()
    expand_align = lev.expandAlign()
    print(expand_align)
    print()


# refwords = "s a5 n || p u5 || ts i̪5 || ərɜ || l i2 ".split()
# hypwords = "s a5 n || p u5 || ts ou2 || t ə1 ".split()
# test_lev_phones1(refwords, hypwords)

# refwords = "ts. ou5 || th iɛ5 n || tsh aiɜ ".split()
# hypwords = "ts. onɡ5 || tɕ iɛ5 n || kh ai5 || k w a5 n ".split()
# test_lev_phones1(refwords, hypwords)

text = ["三不孜儿里",
        "散不走的"]
print(text)
phn = chinese_word_phonemization(text)
test_lev_phones1(phn[0].split(), phn[1].split())

text = ["周天才",
        "中间开关"]
print(text)
phn = chinese_word_phonemization(text)
test_lev_phones1(phn[0].split(), phn[1].split())

text = ["曹师傅",
        "找的时候"]
print(text)
phn = chinese_word_phonemization(text)
test_lev_phones1(phn[0].split(), phn[1].split())

text = ["支付宝到账",
        "知不道仗"]
print(text)
phn = chinese_word_phonemization(text)
test_lev_phones1(phn[0].split(), phn[1].split())


text = ["哦奏是写的周奎奏写的周奎那你几块表还几块表",
        "哦都是写的周都写的周回嘛你几块表还是几块电啊"]
print(text)
phn = chinese_word_phonemization(text)
test_lev_phones1(phn[0].split(), phn[1].split())

text = ["干冒干含有个红灯闪",
        "们们门们们红镇上"]
print(text)
phn = chinese_word_phonemization(text)
test_lev_phones1(phn[0].split(), phn[1].split())