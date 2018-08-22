import pandas as pd
from IPython.display import display, HTML

# Formatting for tables
# Original source: Eric Moyer (https://github.com/epmoyer/ipy_table/issues/24)
def multi_table(table_list):
    ''' Accepts a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' +
        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )

def metadata_count_summary(md):
    """ Given an input dataframe consisting of BelgaLogos metadata, generate a
    summary dataframe counting the number of 'ok' and 'junk' images within the
    dataset, broken down brand-by-brand. The resulting dataframe is arranged to
    mimic the summary tables on the BelgaLogos website."""
    # Perform image counts (total, ok images and junk images)
    total_images = md['brand'].value_counts()
    ok_images    = md[md.ok]['brand'].value_counts()
    junk_images  = md[md.ok == False]['brand'].value_counts()

    # Build a dataframe of image counts from the input dataset
    summary = pd.concat([ok_images, junk_images, total_images], axis=1, sort=True)
    summary.columns = ['#OK', '#Junk', 'Total']
    summary.index.name = 'Logo name'
    return summary
