import sys
import itertools
class FastAreader :
    '''
    Define objects to read FastA files.

    instantiation:
    thisReader = FastAreader ('testTiny.fa')
    usage:
    for head, seq in thisReader.readFasta():
        print (head,seq)
    '''
    def __init__ (self, fname=None):
        '''contructor: saves attribute fname '''
        self.fname = fname

    def doOpen (self):
        ''' Handle file opens, allowing STDIN.'''
        if self.fname is None:
            return sys.stdin
        else:
            return open(self.fname)

    def readFasta (self):
        ''' Read an entire FastA record and return the sequence header/sequence'''
        header = ''
        sequence = ''

        with self.doOpen() as fileH:

            header = ''
            sequence = ''

            # skip to first fasta header
            line = fileH.readline()
            while not line.startswith('>') :
                line = fileH.readline()
            header = line[1:].rstrip()

            for line in fileH:
                if line.startswith ('>'):
                    yield header,sequence
                    header = line[1:].rstrip()
                    sequence = ''
                else :
                    sequence += ''.join(line.rstrip().split()).upper()

        yield header,sequence

'''
Sample motifs from a given sequence - call them reads. Designed for >50k read lengths

for each read:
    - count the frequency of all 4-mers to generate kmer frequency distribution profile
    - add read name and kmer frequency profile to CSV file

'''

# for very large reads, we can probably use smaller samples

class Features:
    def __init__(self, path):
        
        thisReader=FastAreader(path)
        self.heads = []
        self.seqs = []

        for head, seq in thisReader.readFasta():
            self.heads.append(head)
            self.seqs.append(seq)

        # code to generate all permutations of DNA 4-mers
        self.fourMers = [''.join(p) for p in itertools.product('ACGT', repeat=4)]
        

    def sampleMotifs(self, outPath):
        '''
        We have self.fourMers, a list of all possible 4-mers. We also have a list self.heads, and a list self.seqs.
        This method will write to file 'testProfiles.csv' the read name and the kmer frequency profile. The CSV will
        have a column for the read name, and a column for each 4-mer. The values will be the read name and frequency 
        of each 4-mer (with overlap). EG. if the read is "AGTCT", then the 4-mers present are "AGTC", "GTCT".

        for each read, determine the count of all 4-mers.
        '''
        with open('testProfiles.csv', 'w') as file:
            file.write('Read Name,')
            file.write(','.join(self.fourMers))
            file.write('\n')
            for i in range(len(self.heads)):
                read_name = self.heads[i]
                kmer_freq = [str(self.seqs[i].count(kmer) / (len(self.seqs[i])-3)) for kmer in self.fourMers]
                file.write(read_name + ',')
                file.write(','.join(kmer_freq))
                file.write('\n')


def main():
    path = '/Users/jlarbale/Desktop/clusterKdata/codeBase/testReads.txt'
    clusterK = Features(path)
    clusterK.sampleMotifs('')
if __name__ == '__main__':
    main()
