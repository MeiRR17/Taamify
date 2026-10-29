import unittest
from research_os.benchmark_candidates import align_words, transcription_counts


class BenchmarkMetricTests(unittest.TestCase):
    def test_missing_word_is_not_removed_from_denominator(self):
        c=transcription_counts(['וַיָּבֹ֤א','הָאִישׁ֙'],['Mahapakh','Pashta'],'ויב֤א')
        self.assertEqual(c['reference_taamim'],2)
        self.assertEqual(c['correct_word_and_taam'],1)
        self.assertEqual(c['deletions'],1)

    def test_unmarked_word_counts_as_wrong_accent(self):
        c=transcription_counts(['הָאִישׁ֙'],['Pashta'],'האיש')
        self.assertEqual(c['matched_words'],1)
        self.assertEqual(c['correct_word_and_taam'],0)

    def test_insertions_are_visible_and_do_not_shift_later_matches(self):
        c=transcription_counts(['וַיָּבֹ֤א','הָאִישׁ֙'],['Mahapakh','Pashta'],'ויב֤א גם האיש֙')
        self.assertEqual(c['insertions'],1)
        self.assertEqual(c['correct_word_and_taam'],2)
        self.assertEqual(align_words(['א','ב'],['א','ג','ב']),[(0,0),(None,1),(1,2)])


if __name__=='__main__':unittest.main()
