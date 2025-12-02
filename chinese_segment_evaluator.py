from power.levenshtein import Levenshtein, ExpandedAlignment
from phonemizer import phonemize
from phonemizer.separator import Separator
from collections import defaultdict, Counter
import Levenshtein as lev
import re
from chinese_phoneme_aligner import SDAligner


class ChineseSegmentEvaluator:
    def __init__(self, weights=None, reserve_list=None, exclusive_sets=None):
        """
        初始化中文音素区间评估器类
        
        参数:
        weights: 编辑操作的权重字典 (默认: {'C': 0, 'S': 4, 'D': 3, 'I': 3})
        reserve_list: 需要特殊处理的保留标记集合 (默认: {'||', '#'})
        exclusive_sets: 音素分类的排他性集合列表
        """
        self.weights = weights or {'C': 0, 'S': 4, 'D': 3, 'I': 3}
        self.reserve_list = reserve_list or {'||', '#'}
        self.exclusive_sets = exclusive_sets or [
            {'ao', 'ah', 'ow', 'ae', 'er', 'ih', 'eh', 'aa', 'uw', 'ay', 'ax', 'uh', 'aw', 'axr', 'oy', 'ey', 'iy', 'ŋ'},
            {'ts', 'x', 'kh', 'tɕ', 'tsh', 't', 'dx', 'z', 'k', 'm', 'zh', 'y', 'b', 'f', 'eng', 'dh', 'p', 'l', 'r', 'ch', 'em', 'q', 'el', 'd', 'jh', 'sh', 'ng', 'w', 'en', 'g', 'nx', 's', 'hh', 'th', 'v', 'n'},
            {'r', 'axr', 'er'}
        ]
        self.expanded_align = None
        
        # 滚动统计（累计平均）
        self._rolling_count = 0
        self.rolling_recall_avg = 0.0
        self.rolling_precision_avg = 0.0
        self.rolling_f1_avg = 0.0
        self.rolling_cer_avg = 0.0
        self.rolling_sdcer_avg = 0.0
        
        # SDCER 分量比例滚动平均值（记录每种错误类型在SDCER中的比例分量）
        self.rolling_s_phn_ratio_avg = 0.0
        self.rolling_d_phn_ratio_avg = 0.0
        self.rolling_i_phn_ratio_avg = 0.0
        self.rolling_s_chr_ratio_avg = 0.0
        self.rolling_d_chr_ratio_avg = 0.0
        self.rolling_i_chr_ratio_avg = 0.0

        # 供复用的SDAligner实例
        self._sd_aligner = SDAligner(weights=self.weights, reserve_list=self.reserve_list, exclusive_sets=self.exclusive_sets)
    
    
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
        self.expanded_align = lev.expandAlign()#Compact()
        
        return self.expanded_align
    
    
    
    def evaluate_segments(self, ref_text, hyp_text):
        """
        评估带【】标记的中文文本的召回率和准确率
        
        参数:
        ref_text: 参考文本，包含【】,<>标记
        hyp_text: 假设文本，包含【】标记
        
        返回:
        dict: 包含CER, SDCER, 召回率、准确率、交并比等指标的字典
        """

        #1. 获取去掉标点后的文本
        ref_text = re.sub(r'[,，。！？?；：\'"‘’“”]', '', ref_text)
        hyp_text = re.sub(r'[,，。！？?；：\'"‘’“”]', '', hyp_text)
        # 0. 先计算SDAligner提供的指标（基于原始输入）
        ref_plain = ref_text.replace('【', '').replace('】', '')
        hyp_plain = hyp_text.replace('【', '').replace('】', '')

        cer_value = self._sd_aligner.CER(ref_plain, hyp_plain)
        sdcer_result = self._sd_aligner.SDCER(ref_plain, hyp_plain)
        sdcer_value = sdcer_result.get('SDCER', 0.0)

        # if not hyp_plain.strip():

         
        # 2. 获取去掉【】后的纯文本
        ref_text = ref_text.replace('<', '').replace('>', '')
        ref_plain = ref_plain.replace('<', '').replace('>', '')
        # hyp_plain = hyp_text.replace('', '').replace('】', '')
        
        # 3. 执行音素对齐
        self.align_phonemes(ref_plain, hyp_plain)
        
        # 4. 获取对齐结果
        alignment = self.expanded_align.hyp_oriented_alignment()
        ref_tokens = self.expanded_align.s1 + ['||']  # 添加分隔符
        # ['s', 'a5', 'n', '||', 'p', 'u5', '||', 'ts', 'i̪5', '||', 'ərɜ', '||', 'ŋɜ' , '||', 'l', 'i2', '||']
        hyp_tokens = self.expanded_align.s2 + ['||']  # 添加分隔符
        # ['s', 'a5', 'n', '||', 'p', 'u5', '||', 'ts', ''  , ''  , ''   , ''  , 'ou2', '||', 't', 'ə1', '||']
        
        # 5. 构建“共同音素索引空间”：按列遍历(ref_token, hyp_token)，删除空/分隔符列
        def build_common_phoneme_space(ref_tokens, hyp_tokens, separator='||'):
            """
            根据两个对齐的 token 列表（包含分隔符），构建一个共同的、无分隔符的音素空间，
            并计算每个单词在其中的索引跨度 (spans)。

            该函数的核心逻辑是：
            1.  同时遍历 ref_tokens 和 hyp_tokens。
            2.  如果两个 token 中有一个是分隔符，则这对 token 会被忽略，不计入无分隔符列表。
            3.  如果两个 token 都不是分隔符，则它们被添加到各自的无分隔符列表中。
            4.  单词的边界由原始列表中的分隔符决定。每个单词的跨度 (span) 是其在
                无分隔符列表中的 [起始索引, 结束索引)。

            Args:
                ref_tokens (list[str]): 参考 token 列表。
                hyp_tokens (list[str]): 假设 token 列表。两个列表必须等长。
                separator (str, optional): 用于分隔单词的 token。默认为 '||'。

            Returns:
                tuple: 包含四个元素的元组：
                    - ref_phonemes_nosep (list[str]): 移除了分隔符的参考音素列表。
                    - hyp_phonemes_nosep (list[str]): 移除了分隔符的假设音素列表。
                    - ref_char_spans (list[tuple[int, int]]): ref_tokens 中每个单词在
                    ref_phonemes_nosep 中的 (起始, 结束) 索引。
                    - hyp_char_spans (list[tuple[int, int]]): hyp_tokens 中每个单词在
                    hyp_phonemes_nosep 中的 (起始, 结束) 索引。
            """
            if len(ref_tokens) != len(hyp_tokens):
                raise ValueError("输入列表 ref_tokens 和 hyp_tokens 必须等长。")

            ref_phonemes_nosep = []
            hyp_phonemes_nosep = []
            ref_char_spans = []
            hyp_char_spans = []

            # `nosep_idx` 是 phonemes_nosep 列表的当前长度（或下一个元素的索引）
            nosep_idx = 0
            ref_start_idx = 0
            hyp_start_idx = 0

            for ref_tok, hyp_tok in zip(ref_tokens, hyp_tokens):
                # 步骤 1: 检查原始列表中的分隔符，以确定单词边界。
                # 单词的结束位置就是当前 nosep 列表的长度。
                if ref_tok == separator:
                    ref_char_spans.append((ref_start_idx, nosep_idx))
                    ref_start_idx = nosep_idx

                if hyp_tok == separator:
                    hyp_char_spans.append((hyp_start_idx, nosep_idx))
                    hyp_start_idx = nosep_idx

                # 步骤 2: 如果两个 token 都不是分隔符，则将它们添加到 nosep 列表中，
                # 并推进 nosep 索引。
                if ref_tok != separator and hyp_tok != separator:
                    ref_phonemes_nosep.append(ref_tok)
                    hyp_phonemes_nosep.append(hyp_tok)
                    nosep_idx += 1

            return ref_phonemes_nosep, hyp_phonemes_nosep, ref_char_spans, hyp_char_spans

        # 6. 解析【】并得到在纯文本(去掉【】)中的字符区间
        def extract_marked_spans_in_plain(text):
            spans = []  # (start_char, end_char, inner_text)
            plain_pos = 0
            i = 0
            while i < len(text):
                ch = text[i]
                if ch == '【':
                    j = text.find('】', i + 1)
                    if j == -1:
                        # 若不成对，视为普通字符
                        plain_pos += 1
                        i += 1
                        continue
                    inner = text[i + 1:j]
                    length = len(inner)
                    spans.append((plain_pos, plain_pos + length, inner))
                    plain_pos += length
                    i = j + 1
                else:
                    plain_pos += 1
                    i += 1
            return spans

        # 7. 构建音素空间映射与区间
        ref_phonemes_nosep, hyp_phonemes_nosep, ref_char_spans, hyp_char_spans = build_common_phoneme_space(ref_tokens, hyp_tokens)

        ref_marked_plain_spans = extract_marked_spans_in_plain(ref_text)
        hyp_marked_plain_spans = extract_marked_spans_in_plain(hyp_text)

        # 将字符区间映射为音素区间
        def map_char_span_to_phoneme_span(char_spans, start_char, end_char):
            if end_char <= start_char:
                return (0, 0)
            start_ph = char_spans[start_char][0] if start_char < len(char_spans) else (char_spans[-1][1] if char_spans else 0)
            last_char = min(end_char - 1, len(char_spans) - 1)
            end_ph = char_spans[last_char][1] if char_spans else 0
            return (start_ph, end_ph)

        ref_segments_ph = []  # (start_ph, end_ph, text)
        for (cs, ce, seg_txt) in ref_marked_plain_spans:
            ps, pe = map_char_span_to_phoneme_span(ref_char_spans, cs, ce)
            ref_segments_ph.append((ps, pe, seg_txt))

        hyp_segments_ph = []
        for (cs, ce, seg_txt) in hyp_marked_plain_spans:
            ps, pe = map_char_span_to_phoneme_span(hyp_char_spans, cs, ce)
            hyp_segments_ph.append((ps, pe, seg_txt))

        # 8. 计算每对区间的交、并与IoU
        def interval_metrics(a, b):
            as_, ae_ = a
            bs_, be_ = b
            inter = max(0, min(ae_, be_) - max(as_, bs_))
            uni = max(ae_, be_) - min(as_, bs_)
            iou_ = (inter / uni) if uni > 0 else 0.0
            return inter, uni, iou_

        # 9. 基于阈值做一一匹配，统计TP/FP（默认按IoU阈值）
        threshold = 0.5
        candidates = []  # (score, inter, ref_idx, hyp_idx)
        for ri, (rstart, rend, _) in enumerate(ref_segments_ph):
            for hi, (hs, he, _) in enumerate(hyp_segments_ph):
                inter, uni, iou_ = interval_metrics((rstart, rend), (hs, he))
                score = iou_
                candidates.append((score, inter, ri, hi))
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        matched_ref = set()
        matched_hyp = set()
        matches = []  # (ref_idx, hyp_idx, iou, inter)
        for score, inter, ri, hi in candidates:
            if score < threshold:
                break
            if ri in matched_ref or hi in matched_hyp:
                continue
            matched_ref.add(ri)
            matched_hyp.add(hi)
            matches.append((ri, hi, score, inter))

        tp_count = len(matches)
        ref_total = len(ref_segments_ph)
        hyp_total = len(hyp_segments_ph)
        fp_count = max(0, hyp_total - tp_count)
        fn_count = max(0, ref_total - tp_count)
        if ref_total == 0:
            recall = 1.0
            precision = 1.0 if hyp_total == 0 else 0.0
        else:
            precision = (tp_count / hyp_total) if hyp_total else 0.0
            recall = (tp_count / ref_total)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # 10. 汇总全局IoU（并非用于判定，仅用于报告）
        ref_cover = set()
        for (s, e, _) in ref_segments_ph:
            ref_cover.update(range(s, e))
        hyp_cover = set()
        for (s, e, _) in hyp_segments_ph:
            hyp_cover.update(range(s, e))
        intersection = len(ref_cover & hyp_cover)
        union = len(ref_cover | hyp_cover)
        iou = (intersection / union) if union else 0.0
        
        # 更新滚动平均
        self._rolling_count += 1
        
        # 计算当前 SDCER 分量比例
        n_total = sdcer_result.get('N', 1.0)  # 避免除零
        current_s_phn_ratio = sdcer_result.get('S_phn', 0.0) / n_total
        current_d_phn_ratio = sdcer_result.get('D_phn', 0.0) / n_total
        current_i_phn_ratio = sdcer_result.get('I_phn', 0.0) / n_total
        current_s_chr_ratio = sdcer_result.get('S_chr', 0.0) / n_total
        current_d_chr_ratio = sdcer_result.get('D_chr', 0.0) / n_total
        current_i_chr_ratio = sdcer_result.get('I_chr', 0.0) / n_total
        
        # 更新滚动平均值（通过历史平均 × 历史数量 + 当前值）÷ 新数量
        if self._rolling_count == 1:
            # 第一个样本
            self.rolling_recall_avg = recall
            self.rolling_precision_avg = precision
            self.rolling_f1_avg = f1
            self.rolling_cer_avg = cer_value
            self.rolling_sdcer_avg = sdcer_value
            self.rolling_s_phn_ratio_avg = current_s_phn_ratio
            self.rolling_d_phn_ratio_avg = current_d_phn_ratio
            self.rolling_i_phn_ratio_avg = current_i_phn_ratio
            self.rolling_s_chr_ratio_avg = current_s_chr_ratio
            self.rolling_d_chr_ratio_avg = current_d_chr_ratio
            self.rolling_i_chr_ratio_avg = current_i_chr_ratio
        else:
            # 后续样本：新平均 = (历史平均 × 历史数量 + 当前值) ÷ 新数量
            prev_count = self._rolling_count - 1
            self.rolling_recall_avg = (self.rolling_recall_avg * prev_count + recall) / self._rolling_count
            self.rolling_precision_avg = (self.rolling_precision_avg * prev_count + precision) / self._rolling_count
            self.rolling_f1_avg = (self.rolling_f1_avg * prev_count + f1) / self._rolling_count
            self.rolling_cer_avg = (self.rolling_cer_avg * prev_count + cer_value) / self._rolling_count
            self.rolling_sdcer_avg = (self.rolling_sdcer_avg * prev_count + sdcer_value) / self._rolling_count
            self.rolling_s_phn_ratio_avg = (self.rolling_s_phn_ratio_avg * prev_count + current_s_phn_ratio) / self._rolling_count
            self.rolling_d_phn_ratio_avg = (self.rolling_d_phn_ratio_avg * prev_count + current_d_phn_ratio) / self._rolling_count
            self.rolling_i_phn_ratio_avg = (self.rolling_i_phn_ratio_avg * prev_count + current_i_phn_ratio) / self._rolling_count
            self.rolling_s_chr_ratio_avg = (self.rolling_s_chr_ratio_avg * prev_count + current_s_chr_ratio) / self._rolling_count
            self.rolling_d_chr_ratio_avg = (self.rolling_d_chr_ratio_avg * prev_count + current_d_chr_ratio) / self._rolling_count
            self.rolling_i_chr_ratio_avg = (self.rolling_i_chr_ratio_avg * prev_count + current_i_chr_ratio) / self._rolling_count
        
        # 计算衍生分量比例（通过换算得到）
        self.rolling_s_ratio_avg = self.rolling_s_phn_ratio_avg + self.rolling_s_chr_ratio_avg
        self.rolling_d_ratio_avg = self.rolling_d_phn_ratio_avg + self.rolling_d_chr_ratio_avg
        self.rolling_i_ratio_avg = self.rolling_i_phn_ratio_avg + self.rolling_i_chr_ratio_avg
        self.rolling_errors_phn_ratio_avg = self.rolling_s_phn_ratio_avg + self.rolling_d_phn_ratio_avg + self.rolling_i_phn_ratio_avg
        self.rolling_errors_chr_ratio_avg = self.rolling_s_chr_ratio_avg + self.rolling_d_chr_ratio_avg + self.rolling_i_chr_ratio_avg

        # from IPython import embed;embed()
        return {
            'cer': cer_value,
            'sdcer': sdcer_value,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'tp_count': tp_count,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'ref_phn_total': ref_total,
            'hyp_phn_total': hyp_total,
            'iou': iou,
            'intersection': intersection,
            'union': union,
            # SDCER 分量比例（每种错误类型在SDCER中的比例分量）
            's_phn_ratio': sdcer_result.get('S_phn', 0.0) / n_total,
            'd_phn_ratio': sdcer_result.get('D_phn', 0.0) / n_total,
            'i_phn_ratio': sdcer_result.get('I_phn', 0.0) / n_total,
            's_chr_ratio': sdcer_result.get('S_chr', 0.0) / n_total,
            'd_chr_ratio': sdcer_result.get('D_chr', 0.0) / n_total,
            'i_chr_ratio': sdcer_result.get('I_chr', 0.0) / n_total,
            # SDCER 衍生分量比例（通过换算得到）
            's_ratio': (sdcer_result.get('S_phn', 0.0) + sdcer_result.get('S_chr', 0.0)) / n_total,
            'd_ratio': (sdcer_result.get('D_phn', 0.0) + sdcer_result.get('D_chr', 0.0)) / n_total,
            'i_ratio': (sdcer_result.get('I_phn', 0.0) + sdcer_result.get('I_chr', 0.0)) / n_total,
            'n': sdcer_result.get('N', 0.0),
            'errors_phn_ratio': (sdcer_result.get('S_phn', 0.0) + sdcer_result.get('D_phn', 0.0) + sdcer_result.get('I_phn', 0.0)) / n_total,
            'errors_chr_ratio': (sdcer_result.get('S_chr', 0.0) + sdcer_result.get('D_chr', 0.0) + sdcer_result.get('I_chr', 0.0)) / n_total,
            # 保留原始分段信息，便于外部打印
            'ref_phonemes_nosep': ref_phonemes_nosep,
            'hyp_phonemes_nosep': hyp_phonemes_nosep,
            'ref_char_spans':ref_char_spans,
            'hyp_char_spans':hyp_char_spans,
            # 额外返回音素区间的分段
            'ref_segments_phoneme': ref_segments_ph,
            'hyp_segments_phoneme': hyp_segments_ph,
            'matches': matches,
            # 'alignment': alignment,
            'ref_tokens': ref_tokens,
            'hyp_tokens': hyp_tokens,
            # 滚动平均值
            'rolling_recall_avg': self.rolling_recall_avg,
            'rolling_precision_avg': self.rolling_precision_avg,
            'rolling_f1_avg': self.rolling_f1_avg,
            'rolling_cer_avg': self.rolling_cer_avg,
            'rolling_sdcer_avg': self.rolling_sdcer_avg,
            # SDCER 分量比例滚动平均值（基础分量）
            'rolling_s_phn_ratio_avg': self.rolling_s_phn_ratio_avg,
            'rolling_d_phn_ratio_avg': self.rolling_d_phn_ratio_avg,
            'rolling_i_phn_ratio_avg': self.rolling_i_phn_ratio_avg,
            'rolling_s_chr_ratio_avg': self.rolling_s_chr_ratio_avg,
            'rolling_d_chr_ratio_avg': self.rolling_d_chr_ratio_avg,
            'rolling_i_chr_ratio_avg': self.rolling_i_chr_ratio_avg,
            # SDCER 衍生分量比例滚动平均值（通过换算得到）
            'rolling_s_ratio_avg': self.rolling_s_ratio_avg,
            'rolling_d_ratio_avg': self.rolling_d_ratio_avg,
            'rolling_i_ratio_avg': self.rolling_i_ratio_avg,
            'rolling_errors_phn_ratio_avg': self.rolling_errors_phn_ratio_avg,
            'rolling_errors_chr_ratio_avg': self.rolling_errors_chr_ratio_avg,
        }
    
    def _has_overlap(self, start1, end1, start2, end2):
        """
        检查两个区间是否有交集
        
        参数:
        start1, end1: 第一个区间的起始和结束位置
        start2, end2: 第二个区间的起始和结束位置
        
        返回:
        bool: 是否有交集
        """
        return not (end1 <= start2 or end2 <= start1)
    
    
    def print_evaluation_report(self, ref_text, hyp_text):
        """
        打印详细的评估报告
        
        参数:
        ref_text: 参考文本
        hyp_text: 假设文本
        """
        print("=" * 60)
        print("中文音素区间评估报告")
        print("=" * 60)
        print(f"参考文本: {ref_text}")
        print(f"假设文本: {hyp_text}")
        print()
        
        # 执行评估
        result = self.evaluate_segments(ref_text, hyp_text)
        
        # 打印基本指标
        print("评估指标:")
        for k,v in result.items():
            print(k,": ", v)
        print()
        
        # 打印音素对齐信息
        print("音素对齐结果:")
        print(self.expanded_align)
        
        return result


def test_evaluator():
    """测试评估器功能"""
    evaluator = ChineseSegmentEvaluator()
    
    # 测试用例1：你提供的例子
    print("测试用例1:")
    ref_text = "【三不】孜儿嗯里"
    hyp_text = "【散不走】【的】"
    result1 = evaluator.print_evaluation_report(ref_text, hyp_text)
    
    # print("\n" + "="*80 + "\n")
    
    # 测试用例2：更复杂的例子
    print("测试用例2:")
    ref_text = "【三不】孜儿嗯里【测试】文本"
    hyp_text = "【散不走】【的】【测试】文本内容"
    # # ['s', 'a5', 'n', 'p', 'u5', 'ts', 'i̪5', 'ərɜ', 'ŋɜ' , 'l', 'i2', 'tsh', 'o5', 's.', 'i.5', 'w', 'uəɜ', 'n', 'p', 'ə2', 'n', '', '', '', '']
    # # ['s', 'a5', 'n', 'p', 'u5', 'ts', ''  , ''   , 'ou2', 't', 'ə1', 'tsh', 'o5', 's.', 'i.5', 'w', 'uəɜ', 'n', 'p', 'ə2', 'n', 'n', 'ei5', 'ʐ', 'onɡɜ']

    # result2 = evaluator.print_evaluation_report(ref_text, hyp_text)
    
    print("\n" + "="*80 + "\n")
    
    # 测试用例3：没有标记的文本
    # print("测试用例3:")


    result3 = evaluator.print_evaluation_report(ref_text, hyp_text)


if __name__ == "__main__":
    test_evaluator()