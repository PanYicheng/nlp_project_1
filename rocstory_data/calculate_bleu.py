from nltk.translate.bleu_score import sentence_bleu
import sys
if __name__ == '__main__':
    if len(sys.args) < 3:
        print('Usage:\npython calculate_bleu.py reference.txt candidate.txt')
    file_candidate = sys.args[2]
    file_reference = sys.args[1]
    with open(file_candidate, 'rt') as f:
        candidate = f.readlines()
    with open(file_reference, 'rt') as f:
        reference = f.readlines()
    for i, ref in enumerate(reference):
        reference[i] = ref[ref.find('<EOL>')+6:]
    print('{:<20}:'.format('Reference[0]'), reference[0])
    print('{:<20}:'.format('Candidate[0]'), candidate[0])
    bleu = 0
    for i in range(len(reference)):
        bleu += sentence_bleu([reference[i]], candidate[i])
    bleu /= len(reference)
    print(bleu)

