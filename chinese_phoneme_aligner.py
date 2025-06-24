from power.levenshtein import Levenshtein, ExpandedAlignment
from phonemizer import phonemize
from phonemizer.separator import Separator
from collections import defaultdict, Counter

class SDAligner:
    def __init__(self, weights=None, reserve_list=None, exclusive_sets=None):
        """
        初始化SDAligner类
        
        参数:
        weights: 编辑操作的权重字典 (默认: {'C': 0, 'S': 4, 'D': 3, 'I': 3})
        reserve_list: 需要特殊处理的保留标记集合 (默认: {'||', '#'})
        exclusive_sets: 音素分类的排他性集合列表
        """
        self.weights = weights or {'C': 0, 'S': 4, 'D': 3, 'I': 3}
        self.reserve_list = reserve_list or {'||', '#'}
        self.exclusive_sets = exclusive_sets or [
            {'ao', 'ah', 'ow', 'ae', 'er', 'ih', 'eh', 'aa', 'uw', 'ay', 'ax', 'uh', 'aw', 'axr', 'oy', 'ey', 'iy'},
            {'ts', 'x', 'kh', 'tɕ', 'tsh', 't', 'dx', 'z', 'k', 'm', 'zh', 'y', 'b', 'f', 'eng', 'dh', 'p', 'l', 'r', 'ch', 'em', 'q', 'el', 'd', 'jh', 'sh', 'ng', 'w', 'en', 'g', 'nx', 's', 'hh', 'th', 'v', 'n'},
            {'r', 'axr', 'er'}
        ]
        self.expanded_align = None
    
    @staticmethod
    def phonemize_chinese(text):
        """
        将中文文本音素化
        
        参数:
        text: 中文文本字符串列表
        
        返回:
        音素化后的文本列表
        """
        separator = Separator(phone=' ', word=' || ')
        phn = phonemize(
            text,
            language='cmn',
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            separator=separator,
            njobs=4
        )
        return phn

    def align_phonemes(self, ref_str, hyp_str):
        """
        对齐两个中文文本的音素序列
        
        参数:
        ref_str: 参考中文文本
        hyp_str: 假设中文文本
        
        返回:
        扩展对齐对象
        """
        self.ref = ref_str
        self.hyp = hyp_str

        # 音素化中文文本
        ref_phonemes = self.phonemize_chinese([ref_str])[0].split()
        hyp_phonemes = self.phonemize_chinese([hyp_str])[0].split()
        
        # print(f"REF: {' '.join(ref_phonemes)}")
        # print(f"HYP: {' '.join(hyp_phonemes)}")
        
        # 执行对齐
        lev = Levenshtein.align(
            ref_phonemes, 
            hyp_phonemes, 
            lowercase=True,
            weights=self.weights,
            reserve_list=self.reserve_list, 
            exclusive_sets=self.exclusive_sets
        )
        lev.editops()
        self.expanded_align = lev.expandAlign()
        
        # 打印对齐结果
        # eval_str = '  '.join(self.expanded_align.hyp_oriented_alignment())
        # print(f"Eval: {eval_str}")
        # print(self.expanded_align)
        
        return self.expanded_align
    
    def s1(self):
        """获取参考文本的音素标记"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        return self.expanded_align.s1
    
    def s2(self):
        """获取假设文本的音素标记"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        return self.expanded_align.s2
    
    def s1_tokens(self):
        """获取参考文本的音素标记"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        return self.expanded_align.s1_tokens()
    
    def s2_tokens(self):
        """获取假设文本的音素标记"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        return self.expanded_align.s2_tokens()
    
    def hyp_oriented_alignment(self):
        """获取面向假设的对齐序列"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        return self.expanded_align.hyp_oriented_alignment()
    
    def s1_map(self):
        """获取参考文本的映射关系"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        return self.expanded_align.s1_map
    
    def s2_map(self):
        """获取假设文本的映射关系"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        return self.expanded_align.s2_map
    
    # def confusion_pairs(self):
    #     """获取混淆音素对"""
    #     if not self.expanded_align:
    #         raise ValueError("请先执行对齐操作")
    #     return self.expanded_align.confusion_pairs()
    
    def get_alignment_report(self):
        """获取完整的对齐报告"""
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
            
        report = {
            's1': self.s1(),
            's2': self.s2(),
            'alignment': self.hyp_oriented_alignment(),
            's1_map': self.s1_map(),
            's2_map': self.s2_map(),
            'align': self.expanded_align
        }
        # for k in report:
        #     print(f"{k}: \n{report[k]}")
        print(self.ref, self.hyp)
        print(self.expanded_align)
        return report

    def confusion_pairs(self):
        """
        获取汉字级别的混淆对 (Confusion Pairs)。
        
        通过分析音素对齐结果，识别参考文本和假设文本之间在单词（此处为汉字）
        级别上的发音相似但书写不同的错误。这包括一对一替换、删除、插入，以及
        多个字与多个字之间的复杂混淆。

        返回:
            list[tuple[str, str]]: 一个包含混淆对的列表。
                                     每个元组的第一个元素是参考文本中的词，
                                     第二个元素是假设文本中对应的词。
                                     例如对于 "三不孜儿里" vs "散不走的",
                                     可能会返回 [('三', '散'), ('孜儿', '走'), ('里', '的')]
        """
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        
        s1 = self.s1()
        s2 = self.s2()
        ops = self.hyp_oriented_alignment()
        
        # 使用原始文本，以便索引保持一致
        ref_chars = list(self.ref)
        hyp_chars = list(self.hyp)

        # --- 1. 构建音素到汉字索引的映射 ---
        # 映射列表的每个元素，其值表示该音素属于第几个汉字（从0开始）
        ref_phoneme_to_char_map = []
        current_ref_idx = 0
        for token in s1:
            ref_phoneme_to_char_map.append(current_ref_idx)
            if token == '||':
                current_ref_idx += 1
        
        hyp_phoneme_to_char_map = []
        current_hyp_idx = 0
        for token in s2:
            hyp_phoneme_to_char_map.append(current_hyp_idx)
            if token == '||':
                current_hyp_idx += 1

        # --- 2. 识别并合并错误段 ---
        confusion_pairs = []
        in_confusion_segment = False
        segment_start_idx = 0
        i = 0
        
        while i < len(ops):
            op = ops[i]
            token1 = s1[i]
            
            # 确定当前位置是否为“错误”
            is_error = False
            if op != 'C':
                is_error = True
            elif token1 != '||':  # 操作是 'C' 且不是分隔符
                ref_char_idx = ref_phoneme_to_char_map[i]
                hyp_char_idx = hyp_phoneme_to_char_map[i]
                
                # 检查对应的汉字是否不同
                if ref_char_idx < len(ref_chars) and hyp_char_idx < len(hyp_chars):
                    if ref_chars[ref_char_idx] != hyp_chars[hyp_char_idx]:
                        is_error = True
                # 如果一个文本结束了，但另一个还在继续，也算错误
                elif ref_char_idx >= len(ref_chars) or hyp_char_idx >= len(hyp_chars):
                     is_error = True

            if is_error and not in_confusion_segment:
                # 发现一个新的混淆段的开始
                in_confusion_segment = True
                segment_start_idx = i
            elif not is_error and in_confusion_segment:
                # 混淆段结束，处理并提取混淆对
                in_confusion_segment = False
                segment_end_idx = i
                
                ref_indices = {ref_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s1[j] not in ('||', '')}
                hyp_indices = {hyp_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s2[j] not in ('||', '')}

                if ref_indices or hyp_indices:
                    ref_word = "".join(ref_chars[min(ref_indices):max(ref_indices)+1]) if ref_indices else ""
                    hyp_word = "".join(hyp_chars[min(hyp_indices):max(hyp_indices)+1]) if hyp_indices else ""
                    
                    if ref_word or hyp_word:
                        confusion_pairs.append((ref_word, hyp_word))

            i += 1

        # 如果对齐在混淆段中结束，处理最后一个段
        if in_confusion_segment:
            segment_end_idx = len(ops)
            ref_indices = {ref_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s1[j] not in ('||', '')}
            hyp_indices = {hyp_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s2[j] not in ('||', '')}

            if ref_indices or hyp_indices:
                ref_word = "".join(ref_chars[min(ref_indices):max(ref_indices)+1]) if ref_indices else ""
                hyp_word = "".join(hyp_chars[min(hyp_indices):max(hyp_indices)+1]) if hyp_indices else ""
                
                if ref_word or hyp_word:
                    confusion_pairs.append((ref_word, hyp_word))

        return confusion_pairs

    
def test_aligner(ref_text, hyp_text):
    aligner = SDAligner()
    
    # 执行对齐
    expanded_align = aligner.align_phonemes(ref_text, hyp_text)
    
    # 获取对齐报告
    report = aligner.get_alignment_report()
    

    confusion = aligner.confusion_pairs()

    # 输出结果
    print(confusion)
    print()

# 测试代码示例
if __name__ == "__main__":
    ref_text = "三不孜儿里"
    hyp_text = "散不走的"
    test_aligner(ref_text, hyp_text)

    ref_text = "周天才"
    hyp_text = "中间开关"
    test_aligner(ref_text, hyp_text)

    ref_text = "曹师傅"
    hyp_text = "找的时候"
    test_aligner(ref_text, hyp_text)

    ref_text = "支付宝到账"
    hyp_text = "知不道仗"
    test_aligner(ref_text, hyp_text)

    ref_text = "哦奏是写的周奎奏写的周奎那你几块表还几块表"
    hyp_text = "哦都是写的周都写的周回嘛你几块表还是几块电啊"
    test_aligner(ref_text, hyp_text)
    
    