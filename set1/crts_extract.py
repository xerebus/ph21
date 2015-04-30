# Ph21 Set 1
# Aritra Biswas

# crts_extract.py
# Extract magnitude vs. time data from CRTS website in CGI and VOTable
# formats.

import numpy as np
import matplotlib.pyplot as plotter
import StringIO

import urllib
import urllib2
from bs4 import BeautifulSoup
from astropy.io.votable import parse_single_table

def cgi_pull_data(name="Her X1", rad="0.1", out="web"):
    '''Given the name of a system, a radius, and an output format
    "web" or "votable", query the CRTS database and return the page as
    an HTML string.'''

    url = "http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedbid_release2.cgi"
    values = {
        "Name" : name,
        "Rad" : rad,
        "DB" : "photocat", # default database
        "OUT" : out, # defaul to HTML output
        "SHORT" : "short"
    }
    data = urllib.urlencode(values)

    request = urllib2.Request(url, data)
    response = urllib2.urlopen(request)
    page = response.read()

    return page


def html_parse(page):
    '''Given HTML output from CRTS, find the photometry data and
    return it as a 2D list.'''

    # create BeautifulSoup object
    soup = BeautifulSoup(page)

    # find data table by "Photometry of Objs:" caption
    for table in soup.find_all("table"):
        try:
            caption_text = table.tr.caption.string
        except AttributeError:
            caption_text = ""
        if caption_text == "Photometry of Objs:":
            data_table = table
            break

    # construct data array
    pull = lambda row: [float(td.string) for td in row.find_all("td")]
    data = [pull(row) for row in data_table.find_all("tr")]
    data = [row for row in data if row != []] # remove empties

    return np.array(data)


def votable_parse(page):
    '''Given HTML output from CRTS with VOTable requested, find
    the download link, extract the VOTable, and return a 2D array.'''

    # create BeautifulSoup object
    soup = BeautifulSoup(page)
    
    # get href of download link
    for link in soup.find_all("a"):
        if link.string == "download":
            url = link.get("href")

    # get XML output
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    votable_xml = response.read()
    
    # get numpy array - convert string to file for votable function
    table = parse_single_table(StringIO.StringIO(votable_xml))
    data = table.array

    return data


def graph_data(data, type="web"):
    '''Given a 2D array of CRTS data, extracted either from HTML or VOTable,
    graph magnitude vs. time.'''

    if type == "web":
        mag = data[:,1]
        mag_err = data[:,2]
        mjd = data[:,5]
    elif type == "vot":
        # helper function to convert masked array output to simple array
        unmask = lambda ma: [i[0] for i in ma.data]
        mag = unmask(data["Mag"])
        mag_err = unmask(data["Magerr"])
        mjd = unmask(data["ObsTime"])

    plotter.plot(mjd, mag, "r.")
    plotter.errorbar(mjd, mag, yerr = mag_err, fmt = "none", ecolor = "0.25")
    plotter.xlabel("MJD")
    plotter.ylabel("Magnitude")
    plotter.gca().invert_yaxis() # convention for astro magnitude
    plotter.show()


if __name__ == "__main__":
    '''Demonstration.'''

    name = "Her X1"
    rad = "0.1"
    
    page = cgi_pull_data(name, rad, "web")
    data = html_parse(page)
    graph_data(data, "web")

    page = cgi_pull_data(name, rad, "vot")
    data = votable_parse(page)
    graph_data(data, "vot")
