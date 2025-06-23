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
        for k in report:
            print(f"{k}: \n{report[k]}")
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
                                   例如: [('三', '散'), ('孜儿里', '走的')]
        """
        if not self.expanded_align:
            raise ValueError("请先执行 align_phonemes() 进行对齐操作")

        # 移除空格以处理带空格的输入字符串
        ref_chars = list(self.ref.replace(" ", ""))
        hyp_chars = list(self.hyp.replace(" ", ""))
        
        s1_aligned = self.s1_tokens()
        s2_aligned = self.s2_tokens()

        # --- 步骤 1: 生成原始的、逐字的对偶列表 ---
        
        raw_pairs = []
        # 使用 '||' 作为同步点，切分对齐序列
        boundaries = [-1] + [i for i, token in enumerate(s1_aligned) if token == '||']
        
        ref_char_idx = 0
        hyp_char_idx = 0

        for i in range(len(boundaries)):
            start = boundaries[i] + 1
            # 确定分片的结束位置
            end = boundaries[i+1] if i + 1 < len(boundaries) else len(s1_aligned)
            
            if start >= end: # 处理末尾或连续的 '||'
                continue

            s1_segment = s1_aligned[start:end]
            s2_segment = s2_aligned[start:end]

            # 判断该分片是否包含来自REF或HYP的实际音素（非填充符'*'）
            has_ref_phonemes = any(p != '*' for p in s1_segment)
            has_hyp_phonemes = any(p != '*' for p in s2_segment)
            
            ref_word_part = ""
            if has_ref_phonemes:
                if ref_char_idx < len(ref_chars):
                    ref_word_part = ref_chars[ref_char_idx]
                    ref_char_idx += 1
            
            hyp_word_part = ""
            if has_hyp_phonemes:
                if hyp_char_idx < len(hyp_chars):
                    hyp_word_part = hyp_chars[hyp_char_idx]
                    hyp_char_idx += 1
            
            # 只有当该“字槽”至少在一个序列中被占用时才记录
            if ref_word_part or hyp_word_part:
                raw_pairs.append((ref_word_part, hyp_word_part))

        # --- 步骤 2: 合并连续的非正确对偶 ---

        final_pairs = []
        if not raw_pairs:
            return final_pairs

        current_group = []
        for ref_part, hyp_part in raw_pairs:
            # 如果是正确的对齐 (C)，则它是一个断点
            if ref_part == hyp_part and ref_part != "":
                # 如果前面有累积的混淆组，先处理它
                if current_group:
                    ref_combined = "".join(p[0] for p in current_group)
                    hyp_combined = "".join(p[1] for p in current_group)
                    final_pairs.append((ref_combined, hyp_combined))
                    current_group = []
            # 如果是错误 (S, D, I) 或空对齐，则加入当前混淆组
            else:
                current_group.append((ref_part, hyp_part))

        # 处理循环结束后可能遗留的最后一个混淆组
        if current_group:
            ref_combined = "".join(p[0] for p in current_group)
            hyp_combined = "".join(p[1] for p in current_group)
            final_pairs.append((ref_combined, hyp_combined))
            
        return final_pairs

    


# 测试代码示例
if __name__ == "__main__":
    # ref_text = "三不孜儿里"
    # hyp_text = "散不走的"
    ref_text = "周天才"
    hyp_text = "中间开关"

    ref_text = "曹师傅"
    hyp_text = "找的时候"

    ref_text = "支付宝到账"
    hyp_text = "知不道仗"

    ref_text = "哦奏是写的周奎奏写的周奎那你几块表还几块表"
    hyp_text = "哦都是写的周都写的周回嘛你几块表还是几块电啊"
    
    aligner = SDAligner()
    
    # 执行对齐
    expanded_align = aligner.align_phonemes(ref_text, hyp_text)
    
    # 获取对齐报告
    report = aligner.get_alignment_report()
    

    confusion = aligner.confusion_pairs()

    # 输出结果
    print(confusion)