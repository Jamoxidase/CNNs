import random
import sys


class FastAreader:
    '''
    Define objects to read FastA files.

    instantiation:
    thisReader = FastAreader ('testTiny.fa')
    usage:
    for head, seq in thisReader.readFasta():
        print (head,seq)
    '''

    def __init__(self, fname=None):
        '''contructor: saves attribute fname '''
        self.fname = fname

    def doOpen(self):
        ''' Handle file opens, allowing STDIN.'''
        if self.fname is None:
            return sys.stdin
        else:
            return open(self.fname)

    def readFasta(self):
        ''' Read an entire FastA record and return the sequence header/sequence'''
        header = ''
        sequence = ''

        with self.doOpen() as fileH:

            header = ''
            sequence = ''

            # skip to first fasta header
            line = fileH.readline()
            while not line.startswith('>'):
                line = fileH.readline()
            header = line[1:].rstrip()

            for line in fileH:
                if line.startswith('>'):
                    yield header, sequence
                    header = line[1:].rstrip()
                    sequence = ''
                else:
                    sequence += ''.join(line.rstrip().split()).upper()

        yield header, sequence



class ReadMaker:
    def __init__(self, genome_paths, coverage, lenRange):
        self.head = []
        self.seq = []
        for path in genome_paths:
            head, seq = self.read_genome(path)
            self.head.append(head)
            self.seq.append(seq)

        self.coverage = coverage
        self.lenRange = lenRange

    def read_genome(self, genome_path):
        thisReader = FastAreader(genome_path)

        for head, seq in thisReader.readFasta():
            head = head.replace(',', '_')
            return head, seq

    def make_reads(self):
        '''
        generate reads from the genome

        Determine read length with random.randint between s, l (lenRange) and extract read from the begaining of genome,
        then remove that read from the genome. Repeat until genome is empty. When the readlength is greater than the 
        remaining genome length, extract the remaining genome as the last read. 

        Reades should be directly written/added to a file (called testReads.txt) in fasta format (read name given on a line begainning with >, 
        sequence beneath.) Each read should be given a unique name, by appending a number to the base name for the genome.

        '''
        counter = 0
        s, l = self.lenRange
        # head = self.head # base name for genome

        num_genomes = len(self.seq)

        with open('testReads.txt', 'w') as f:
            for num in range(num_genomes):  # for each genome

                for i in range(self.coverage):
                    # raw python string of genome sequence
                    genome = self.seq[num]
                    head = self.head[num]
                    while genome:
                        read_len = random.randint(s, l)
                        if read_len > len(genome):
                            read_len = len(genome)
                        read = genome[:read_len]
                        counter += 1
                        f.write(f'>{head}_{counter}_{len(read)}\n{read}\n')
                        genome = genome[read_len:]


def main():
    genome_paths = ['/Users/jlarbale/Desktop/clusterKdata/Genomes/Assembly/EC_GCF_ASM584v2_genomic.txt', 
                    '/Users/jlarbale/Desktop/clusterKdata/Genomes/Assembly/MC_GCF_001578075.1_ASM157807v1_genomic.txt',
                    '/Users/jlarbale/Desktop/clusterKdata/Genomes/Assembly/GCF_000317695.1_ASM31769v1_genomic.txt', 
                    '/Users/jlarbale/Desktop/clusterKdata/Genomes/Assembly/GCF_000146045.2_R64_genomic.txt']
    coverage = 30
    lenRange = (16000, 50000)
    readMaker = ReadMaker(genome_paths, coverage, lenRange)
    readMaker.make_reads()


if __name__ == '__main__':
    main()
