

def read_csv(file,has_columns=True,delimiter=','):
  i_f = open(file,'r')
  
  if has_columns:
    line = i_f.readline()
    columns = line.strip().split(delimiter)

  for line in i_f:
    items = line.strip().split(delimiter)
    row = {}
    for ind in range(len(items)):
      column = columns[ind] if has_columns else ind
      row[column] = items[ind]

    yield row

  i_f.close()