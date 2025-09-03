from power.levenshtein import Levenshtein, ExpandedAlignment
from phonemizer import phonemize
from phonemizer.separator import Separator
from collections import defaultdict, Counter
import Levenshtein as lev

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
        print(self.ref)
        print(self.hyp)
        print(self.expanded_align)
        
        return report

    def confusion_pairs(self, include_correct=False):
        """
        获取汉字级别的混淆对 (Confusion Pairs)。
        如果include_correct=True，则返回所有对齐段，包括正确对齐的部分。
        返回:
            list[tuple[str, str, bool]]: (ref_word, hyp_word, is_correct)
        """
        if not self.expanded_align:
            raise ValueError("请先执行对齐操作")
        s1 = self.s1()
        s2 = self.s2()
        ops = self.hyp_oriented_alignment()
        ref_chars = list(self.ref)
        hyp_chars = list(self.hyp)
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
        pairs = []
        in_segment = False
        segment_start_idx = 0
        segment_is_error = False
        i = 0
        while i < len(ops):
            op = ops[i]
            token1 = s1[i]
            is_error = False
            if op != 'C':
                is_error = True
            elif token1 != '||':
                ref_char_idx = ref_phoneme_to_char_map[i]
                hyp_char_idx = hyp_phoneme_to_char_map[i]
                if ref_char_idx < len(ref_chars) and hyp_char_idx < len(hyp_chars):
                    if ref_chars[ref_char_idx] != hyp_chars[hyp_char_idx]:
                        is_error = True
                elif ref_char_idx >= len(ref_chars) or hyp_char_idx >= len(hyp_chars):
                    is_error = True
            if not in_segment:
                in_segment = True
                segment_start_idx = i
                segment_is_error = is_error
            elif is_error != segment_is_error:
                # 段落类型发生变化，结束前一段
                segment_end_idx = i
                ref_indices = {ref_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s1[j] not in ('||', '')}
                hyp_indices = {hyp_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s2[j] not in ('||', '')}
                ref_word = "".join(ref_chars[min(ref_indices):max(ref_indices)+1]) if ref_indices else ""
                hyp_word = "".join(hyp_chars[min(hyp_indices):max(hyp_indices)+1]) if hyp_indices else ""
                if include_correct or segment_is_error:
                    pairs.append((ref_word, hyp_word, not segment_is_error))
                segment_start_idx = i
                segment_is_error = is_error
            i += 1
        # 处理最后一段
        if in_segment:
            segment_end_idx = len(ops)
            ref_indices = {ref_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s1[j] not in ('||', '')}
            hyp_indices = {hyp_phoneme_to_char_map[j] for j in range(segment_start_idx, segment_end_idx) if s2[j] not in ('||', '')}
            ref_word = "".join(ref_chars[min(ref_indices):max(ref_indices)+1]) if ref_indices else ""
            hyp_word = "".join(hyp_chars[min(hyp_indices):max(hyp_indices)+1]) if hyp_indices else ""
            if include_correct or segment_is_error:
                pairs.append((ref_word, hyp_word, not segment_is_error))
        return pairs
    
    def _parse_segments(self, ref):
        """
        将ref分割为带括号和不带括号的片段，返回[(is_phoneme, text)]
        is_phoneme: True表示<>内，False表示普通汉字
        """
        import re
        segments = []
        last = 0
        for m in re.finditer(r'<([^>]*)>', ref):
            if m.start() > last:
                # 前面有普通汉字
                segments.append((False, ref[last:m.start()]))
            segments.append((True, m.group(1)))
            last = m.end()
        if last < len(ref):
            segments.append((False, ref[last:]))
        return segments

    def SDCER(self, ref, hyp):
        """
        计算Segment-Dependent CER，遍历每个confusion_pair，判断ref片段是否在<>中，选择按音素或汉字计算，并区分S, D, I三种错误。
        音素片段按'||'分割，每个音节合并为一个整体。使用Levenshtein.editops严格统计S/D/I。
        """
        # 1. 解析ref的分段区间
        segments = self._parse_segments(ref)
        seg_spans = []  # [(is_phoneme, start, end, text)]
        idx = 0
        for is_phoneme, text in segments:
            if not text:
                continue
            start = idx
            end = idx + len(text)
            seg_spans.append((is_phoneme, start, end, text))
            idx = end
        # 2. 对齐
        # print(seg_spans)
        ref_plain = ref.replace('<','').replace('>','')
        self.align_phonemes(ref_plain, hyp)
        confusion_pairs = self.confusion_pairs(include_correct=True)
        # print(self.expanded_align)
        # print(confusion_pairs)
        # 3. 遍历每个pair，判断ref_word属于哪个片段
        total = len(ref_plain)
        # 分别统计音素区间与汉字区间的S/D/I（仅维护分量，最终再汇总）
        S_phn = D_phn = I_phn = 0.0
        S_chr = D_chr = I_chr = 0.0
        ref_idx = 0
        for ref_word, hyp_word, _ in confusion_pairs:
            # 找到ref_word在ref_plain中的起止
            if not ref_word:
                # 插入错误
                I_chr += len(hyp_word)
                # total += len(hyp_word)
                continue
            if not hyp_word:
                # 删除错误
                D_chr += len(ref_word)
                # total += len(ref_word)
                ref_idx += len(ref_word)
                continue
            # ref_word非空，找到它属于哪个片段
            ref_start = ref_idx
            ref_end = ref_idx + len(ref_word)
            # 找到对应的seg
            for is_phoneme, seg_start, seg_end, seg_text in seg_spans:
                if ref_start >= seg_start and ref_end <= seg_end:
                    break
            if is_phoneme:
                # UDS区间：音素比对，按'||'分割，每个音节合并为一个整体
                # 根据SDCER公式，需要将音素级错误归一化到字符级：e_i * (c_i/p_i)
                def get_syllables(text):
                    phn = self.phonemize_chinese([text])[0]
                    sylls = phn.replace('||', ' ').split(' ') #.replace(' ', '')
                    return [s.strip() for s in sylls if s.strip()]
                ref_sylls = get_syllables(ref_word)
                hyp_sylls = get_syllables(hyp_word) if hyp_word else []
                # print("!! ", ref_sylls, hyp_sylls, ref_sylls == hyp_sylls)
                
                # 计算字符数和音素数
                c_i = len(ref_word)  # 字符数
                p_i = len(ref_sylls)  # 音素数
                
                if ref_sylls == hyp_sylls:
                    pass  # 全对，无错误
                else:
                    ops = lev.editops(ref_sylls, hyp_sylls)
                    # 根据SDCER公式，将音素级错误归一化到字符级：e_i * (c_i/p_i)
                    normalization_factor = c_i / p_i if p_i > 0 else 1.0
                    
                    # 按错误类型统计，每个错误都乘以归一化因子
                    for op, i, j in ops:
                        if op == 'replace':
                            S_phn += 1 * normalization_factor
                        elif op == 'delete':
                            D_phn += 1 * normalization_factor
                        elif op == 'insert':
                            I_phn += 1 * normalization_factor
            else:
                # 汉字比对
                # total += len(ref_word)
                ops = lev.editops(ref_word, hyp_word)
                for op, i, j in ops:
                    if op == 'replace':
                        S_chr += 1
                    elif op == 'delete':
                        D_chr += 1
                    elif op == 'insert':
                        I_chr += 1
            ref_idx += len(ref_word)
            # print(ref_word, hyp_word, S, D, I)
            # import pdb; pdb.set_trace()
        # 汇总总量
        S = S_phn + S_chr
        D = D_phn + D_chr
        I = I_phn + I_chr
        errors = S + D + I
        errors_phn = S_phn + D_phn + I_phn
        errors_chr = S_chr + D_chr + I_chr
        return {
            'SDCER': errors / total if total > 0 else 0,
            'S': S, 'D': D, 'I': I, 'N': total,
            'S_phn': S_phn, 'D_phn': D_phn, 'I_phn': I_phn,
            'S_chr': S_chr, 'D_chr': D_chr, 'I_chr': I_chr,
            'errors_phn': errors_phn, 'errors_chr': errors_chr
        }

    def CER(self, ref, hyp):
        """
        计算CER
        """
        ref_plain = ref.replace('<','').replace('>','')
        return lev.ratio(ref_plain, hyp)


    
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

def test_sdcer(ref_text, hyp_text):
    aligner = SDAligner()
    
    sdcer = aligner.SDCER(ref_text, hyp_text)
    ref_plain = ref_text.replace('<','').replace('>','')
    ops = lev.editops(ref_plain, hyp_text)

    # 输出结果
    print(sdcer)
    print(len(ops)/len(ref_plain))
    print()

# 测试代码示例
if __name__ == "__main__":
    # ref_text = ""
    # hyp_text = ""
    # test_aligner(ref_text, hyp_text)

    # ref_text = "<三不孜儿嗯>里"
    ref_text = "三不孜儿嗯里"
    hyp_text = "散不走的"
    test_aligner(ref_text, hyp_text)
    # test_sdcer(ref_text, hyp_text)

    ref_text = "你<散不子儿>地看哈短信息"
    hyp_text = "你散不子儿地看哈端信席"
    test_sdcer(ref_text, hyp_text)

    ref_text = "你<散不子儿>地看哈短信息"
    hyp_text = "你三不走地看哈短信息"
    test_sdcer(ref_text, hyp_text)
    
    ref_text = "哎<蒽>抽个空过来把<哥>电费交<葭>啥"
    hyp_text = "唉䅰抽个孔过来把歌电费交下哈"
    # test_aligner(ref_text, hyp_text)
    test_sdcer(ref_text, hyp_text)


    ref_text = "我坎儿问哈子那个"
    hyp_text = "我想要玩这个"
    test_aligner(ref_text, hyp_text)
    # ref_text = "周天才"
    # hyp_text = "中间开关"
    # test_aligner(ref_text, hyp_text)

    # ref_text = "曹师傅"
    # hyp_text = "找的时候"
    # test_aligner(ref_text, hyp_text)

    # ref_text = "支付宝到账"
    # hyp_text = "知不道仗"
    # test_aligner(ref_text, hyp_text)

    # ref_text = "哦奏是写的周奎奏写的周奎那你几块表还几块表"
    # hyp_text = "哦都是写的周都写的周回嘛你几块表还是几块电啊"
    # test_aligner(ref_text, hyp_text)

    # ref_text = "你个老群呃我也这这这喝喝两吨我都息日"
    # hyp_text = "你个老熊了我现在这这这活活两顿了我不行"
    # test_aligner(ref_text, hyp_text)

    # ref_text = "我不是说在你这买里个阿莫西林门不是"
    # hyp_text = "我不是那两百个人不搞都行你们再出来"
    # test_aligner(ref_text, hyp_text)

    # ref_text = "阿莫西林在你哪看可"
    # hyp_text = "我不行你了给他发扣了"
    # test_aligner(ref_text, hyp_text)

    
    
    