from IPython.display import display, HTML

# Formatting for tables
# Original source: Eric Moyer (https://github.com/epmoyer/ipy_table/issues/24)
def multi_table(table_list):
    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' +
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )
