import unicodecsv as csv
import json
import os

import nltk
from pattern3 import en
import spacy


mnli_dir = os.path.expanduser('../../dataset/snli')
mnli_train = os.path.join(mnli_dir, 'snli_1.0_train.jsonl')
mnli_headers = ['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'gold_label']

nlp = spacy.load('en_core_web_sm')
ner = nlp.create_pipe('ner')
parser = nlp.create_pipe('parser')
lemmatizer = nltk.stem.WordNetLemmatizer()


def lower_first(s):
    return s[0].lower() + s[1:]        

def upper_first(s):
    return s[0].upper() + s[1:]        


class MNLISyntacticRegularizer(object):

    def __init__(self):
        self.present_to_past = {}
        self.present_to_vb = {}
        self.present_to_vbz = {}

    def tsv(self, filename):
        return csv.writer(open(filename, 'wb'), delimiter='\t', encoding='utf-8')

    def loop(self, debug=False):
        w_inv_orig = self.tsv('inv_orig.tsv')
        w_inv_trsf = self.tsv('inv_trsf.tsv')
        w_pass_orig = self.tsv('pass_orig.tsv')
        w_pass_trsf = self.tsv('pass_trsf.tsv')

        self.lines = open(mnli_train).readlines()
        #with open(mnli_train) as f:
         #   self.lines = json.load(f)
        #import pdb; pdb.set_trace()
        already_seen = set()
        self.dicts = []
        n = 0
        for i, line in enumerate(self.lines):
            j = json.loads(line)
            
            self.dicts.append(j)
            if i % 10000 == 0: # 10000
                print('%d out of %d' % (i, len(self.lines)))
            if debug and i == 10000: # 10000
                break
            # if j['genre'] == 'telephone':
            #     continue

            tree = j['hyptree'] = nltk.tree.Tree.fromstring(j['sentence2_parse'])

            ss = [x for x in tree.subtrees() if x.label() == 'S']
            

            for s in ss[:1]:
                if len(s) < 2:  # Not a full NP + VP sentence
                    continue

                subj_head = self.get_np_head(s[0])
                if subj_head is None:
                    continue
                subject_number = self.get_np_number(s[0])

                k = 1            
                while (s[k].label() not in (u'VP', u'SBAR', u'ADJP')) and (k < len(s) - 1):
                    k+=1

                if k == len(s) - 1:
                    continue
		#iterate through top level branches to find VP
                vp_head = self.get_vp_head(s[k])
                if vp_head[0] is None:
                    continue

                subj = ' '.join(s[0].flatten())
                arguments = tuple(x.label() for x in s[1][1:])

                if (arguments != ('NP',) or 
                        en.lemma(vp_head[0]) in ['be', 'have']):
                    continue		

                direct_object = ' '.join(s[1][1].flatten()) 
                object_number = self.get_np_number(s[1][1])
                if object_number is None:
                    # Personal pronoun, very complex NP, or parse error
                    continue
                lookup = en.tenses(vp_head[0])

                if len(lookup) == 0:
                    if vp_head[0][-2:]:
                        tense = en.PAST
                    else:
                        tense = en.PRESENT
                else:
                    if en.tenses(vp_head[0])[0][0] == u'past':
                        tense = en.PAST
                    else:
                        tense = en.PRESENT

                subjobj_rev_hyp = ' '.join([
                    upper_first(direct_object),
                    #keep tense
                    en.conjugate(vp_head[0], number=object_number, tense = tense),
                    lower_first(subj)]) + '.'

                passive_hyp_same_meaning = ' '.join([
                    upper_first(direct_object),
                    self.passivize_vp(s[k], object_number),
                    lower_first(subj)]) + '.'

                passive_hyp_inverted = ' '.join([
                    subj,
                    self.passivize_vp(s[k], subject_number),
                    direct_object]) + '.'

                if j['gold_label'] == 'entailment':
                    self.mnli_row(w_inv_orig, 1000000 + n, # 1000000
                            j['sentence1'], subjobj_rev_hyp, 'neutral')

                self.mnli_row(w_inv_trsf, 1000000 + n, # 1000000
                        j['sentence2'], subjobj_rev_hyp, 'neutral')

                self.mnli_row(w_pass_orig, 1000000 + n, # 1000000
                        j['sentence1'], passive_hyp_same_meaning, 
                        j['gold_label'])

                self.mnli_row(w_pass_trsf, 1000000 + n, # 1000000
                        j['sentence2'], passive_hyp_inverted, 'neutral')
                self.mnli_row(w_pass_trsf, 2000000 + n, # 2000000
                        j['sentence2'], passive_hyp_same_meaning, 'entailment')

                n += 1

    def mnli_row(self, writer, i, premise, hypothesis, label):
        row = [str(i)] + ['ba'] * 7 + [premise, hypothesis, 'ba', label]
        writer.writerow(row)

    def get_vp_head(self, vp):
        head = None
        if vp.label() == 'VP':
            while True:
                nested_vps = [x for x in vp[1:] if x.label() == 'VP']
                if len(nested_vps) == 0:
                    break
                vp = nested_vps[0]
            if vp[0].label().startswith('VB'):
                head = vp[0][0].lower()

        return (head, vp[0].label())

    def passivize_vp(self, vp, subj_num=en.SINGULAR):
        head = None
        flat = vp.flatten()
        if vp.label() == 'VP':
            nesters = []
            while True:
                nesters.append(vp[0][0])
                nested_vps = [x for x in vp[1:] if x.label() == 'VP']
                if len(nested_vps) == 0:
                    break
                vp = nested_vps[0]
            label = vp[0].label()
            if label.startswith('VB'):
                head = vp[0][0].lower()
                if len(nesters) > 1:
                    passivizer = 'be'
                elif label in ['VBP', 'VB', 'VBZ']: 
                    # 'VB' here (not nested) is a POS tag error
                    passivizer = 'are' if subj_num == en.PLURAL else 'is'
                elif label == 'VBD' or label == 'VBN':
                    # 'VBN' here (not nested) is a POS tag error
                    passivizer = 'were' if subj_num == en.PLURAL else 'was'
                    # Alternatively, figure out the number of the subject
                    # to decide whether it's was or were
                else:
                    passivizer = 'is'
                vbn = en.conjugate(head, 'ppart')

        return '%s %s by' % (passivizer, vbn)

    def get_np_head(self, np):
        head = None
        if (np.label() == 'NP' and np[0].label() == 'DT'):
            head_candidates = [x for x in np[1:] if
                    x.label().startswith('NN')]
            if len(head_candidates) == 1:
                # > 1: Complex noun phrases unlikely to be useful
                # 0: Pronominal subjects like "many"
                head = lemmatizer.lemmatize(head_candidates[0][0])
        return head 

    def get_np_number(self, np):
        number = None
        if np[0].label() == 'NP':
            np = np[0]
        head_candidates = [x for x in np if x.label().startswith('NN')]
        if len(head_candidates) == 1:
            label = head_candidates[0].label()
            number = en.PLURAL if label == 'NNS' else en.SINGULAR
        elif len(head_candidates) > 1:
            number = en.PLURAL
        return number 

regularizer = MNLISyntacticRegularizer()
regularizer.loop()