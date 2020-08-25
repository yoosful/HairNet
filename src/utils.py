import struct

def save_binary(strands, filename):
    n_strand = strands.shape[2]*strands.shape[3]
    n_vertex = strands.shape[0]
    fmt = 'i'
    data = [n_strand]
    for i in range(strands.shape[2]):
        for j in range(strands.shape[3]):
            fmt += 'i'
            data += [n_vertex]
            for k in range(strands.shape[0]): # vertex (100)
                fmt += 'fff'
                data += strands[k,:3,i,j].tolist()
    with open (filename, 'wb') as wf:
        wf.write(struct.pack(fmt, *data))