# This modules declares a mini language to filter files from a corpus
# on the basis of properties of their XML content

# At the moment the module assumes we are working with xml.etree.ElementTree


def parse(expr):
    """This function parses an expression in the selection language defined here
       and returns an object of type DOMFilter (or one of its subclasses most likely)"""

    return eval(expr,__dom_filter_parse_globals,__dom_filter_parse_locals)

# Base class

class DOMFilter(object):
    """Generic class that actually doesn't filter anything"""

    def accept(self,dom):
        """This function should be redefined by subclasses to actually perform tests on the DOM object

           It should return True in case the DOM object satisfies the constraints encoded by the class and False otherwise.
        """
        return True

    def __add__(self,other):
        return DisjunctionFilter(self,other)

    def __mul__(self,other):
        return ConjunctionFilter(self,other)

    def __neg__(self):
        return NegationFilter(self)


# Logical combinators


class ConjunctionFilter(DOMFilter):
    """Class that represent the conjunction of two filters"""

    def __init__(self,filter1,filter2):
        self.filter1 = filter1
        self.filter2 = filter2

    def accept(self,dom):
        return self.filter1.accept(dom) and self.filter2.accept(dom)


class DisjunctionFilter(DOMFilter):
    """Class that represent the disjunction of two filters"""

    def __init__(self,filter1,filter2):
        self.filter1 = filter1
        self.filter2 = filter2

    def accept(self,dom):
        return self.filter1.accept(dom) or self.filter2.accept(dom)

    
class NegationFilter(DOMFilter):
    """Class that represent the negation of another filter"""

    def __init__(self,filter):
        self.filter = filter

    def accept(self,dom):
        return not self.filter.accept(dom)


# Atomic filters

class TextClassificationFilter(DOMFilter):
    """Class to check for a single category assigned to the text

       Most of the codes are targets attribute of the <catRef> element in the TEI header of the file,
       and that's where the class looks for them.
       Oddly the following codes are NOT there:

       (these are values of the type attribute of <wtext>)
       ACPROSE
       FICTION
       NEWS
       NONAC
       OTHERPUB
       UNPUB

       (these are values of the type attribute of <stext>)
       CONVRSN
       OTHERSP

       The class is smart enough to figure out where to look for each code
       so we don't need different classes for these cases
    """

    def __init__(self,code):
        self.code = code
        if self.code in ['ACPROSE','FICTION','NEWS','NONAC','OTHERPUB','UNPUB']:
            self.accept = self.accept_wtext
        elif self.code in ['CONVRSN','OTHERSP']:
            self.accept = self.accept_stext
        else:
            self.accept = self.accept_tei

    def accept_wtext(self,dom):
        wtexts = dom.findall('.//wtext')
        if len(wtexts) == 0:
            return False
        else:
            return self.code == wtexts[0].attrib.get('type')

    def accept_stext(self,dom):
        stexts = dom.findall('.//stext')
        if len(stexts) == 0:
            return False
        else:
            return self.code == stexts[0].attrib.get('type')

    def accept_tei(self,dom):
        cat_refs = dom.findall('.//textClass/catRef') # there should be only one actually
        if len(cat_refs) == 0:
            return False
        else:
            codes = cat_refs[0].attrib.get('targets').split()
            return self.code in codes

class ElementPresenceFilter(DOMFilter):
    """ Class to check the presence of a certain element in the XML"""

    def __init__(self,elementName):
        self.elementName = elementName

    def accept(self,dom):
        return len(dom.findall('.//' + self.elementName)) > 0


# Predefined filters

# Hand crafted filters

POETRY = ElementPresenceFilter('lg') # rough but it should work

# Automatically generated from xgrammar.xml
# Written
WRI = TextClassificationFilter('WRI')

# Transcribed speech
SPO = TextClassificationFilter('SPO')

# Academic prose
ACPROSE = TextClassificationFilter('ACPROSE')

# Fiction and verse
FICTION = TextClassificationFilter('FICTION')

# Non-academic prose and biography
NONAC = TextClassificationFilter('NONAC')

# Newspapers
NEWS = TextClassificationFilter('NEWS')

# Other published written material
OTHERPUB = TextClassificationFilter('OTHERPUB')

# Unpublished written material
UNPUB = TextClassificationFilter('UNPUB')

# Spoken conversation
CONVRSN = TextClassificationFilter('CONVRSN')

# Other spoken material
OTHERSP = TextClassificationFilter('OTHERSP')

# Ownership unclaimed
ALLAVA0 = TextClassificationFilter('ALLAVA0')

# Worldwide rights cleared
ALLAVA2 = TextClassificationFilter('ALLAVA2')

# Spoken demographic
ALLTYP1 = TextClassificationFilter('ALLTYP1')

# Spoken context-governed
ALLTYP2 = TextClassificationFilter('ALLTYP2')

# Written books and periodicals
ALLTYP3 = TextClassificationFilter('ALLTYP3')

# Written-to-be-spoken
ALLTYP4 = TextClassificationFilter('ALLTYP4')

# Written miscellaneous
ALLTYP5 = TextClassificationFilter('ALLTYP5')

# Educational/Informative
SCGDOM1 = TextClassificationFilter('SCGDOM1')

# Business
SCGDOM2 = TextClassificationFilter('SCGDOM2')

# Public/Institutional
SCGDOM3 = TextClassificationFilter('SCGDOM3')

# Leisure
SCGDOM4 = TextClassificationFilter('SCGDOM4')

# Respondent Age 0-14
SDEAGE1 = TextClassificationFilter('SDEAGE1')

# Respondent Age 15-24
SDEAGE2 = TextClassificationFilter('SDEAGE2')

# Respondent Age 25-34
SDEAGE3 = TextClassificationFilter('SDEAGE3')

# Respondent Age 35-44
SDEAGE4 = TextClassificationFilter('SDEAGE4')

# Respondent Age 45-59
SDEAGE5 = TextClassificationFilter('SDEAGE5')

# Respondent Age 60+
SDEAGE6 = TextClassificationFilter('SDEAGE6')

# Unknown class
SDECLA0 = TextClassificationFilter('SDECLA0')

# AB respondent
SDECLA1 = TextClassificationFilter('SDECLA1')

# C1 respondent
SDECLA2 = TextClassificationFilter('SDECLA2')

# C2 respondent
SDECLA3 = TextClassificationFilter('SDECLA3')

# DE respondent
SDECLA4 = TextClassificationFilter('SDECLA4')

# Respondent sex unknown
SDESEX0 = TextClassificationFilter('SDESEX0')

# Male respondent
SDESEX1 = TextClassificationFilter('SDESEX1')

# Female respondent
SDESEX2 = TextClassificationFilter('SDESEX2')

# Monologue
SPOLOG1 = TextClassificationFilter('SPOLOG1')

# Dialogue
SPOLOG2 = TextClassificationFilter('SPOLOG2')

# Unknown
SPOREG0 = TextClassificationFilter('SPOREG0')

# South
SPOREG1 = TextClassificationFilter('SPOREG1')

# Midlands
SPOREG2 = TextClassificationFilter('SPOREG2')

# North
SPOREG3 = TextClassificationFilter('SPOREG3')

# Child audience
WRIAUD1 = TextClassificationFilter('WRIAUD1')

# Teenager audience
WRIAUD2 = TextClassificationFilter('WRIAUD2')

# Adult audience
WRIAUD3 = TextClassificationFilter('WRIAUD3')

# Any audience
WRIAUD4 = TextClassificationFilter('WRIAUD4')

# Male audience
WRITAS1 = TextClassificationFilter('WRITAS1')

# Female audience
WRITAS2 = TextClassificationFilter('WRITAS2')

# Mixed audience
WRITAS3 = TextClassificationFilter('WRITAS3')

# Unknown audience
WRITAS0 = TextClassificationFilter('WRITAS0')

# Author age unknown
WRIAAG0 = TextClassificationFilter('WRIAAG0')

# Author age 0-14
WRIAAG1 = TextClassificationFilter('WRIAAG1')

# Author age 15-24
WRIAAG2 = TextClassificationFilter('WRIAAG2')

# Author age 25-34
WRIAAG3 = TextClassificationFilter('WRIAAG3')

# Author age 35-44
WRIAAG4 = TextClassificationFilter('WRIAAG4')

# Author age 45-59
WRIAAG5 = TextClassificationFilter('WRIAAG5')

# Author age 60+
WRIAAG6 = TextClassificationFilter('WRIAAG6')

# Author domicile UK and Ireland
WRIAD1 = TextClassificationFilter('WRIAD1')

# Author domicile Commonwealth
WRIAD2 = TextClassificationFilter('WRIAD2')

# Author domicile Continental Europe
WRIAD3 = TextClassificationFilter('WRIAD3')

# Author domicile USA
WRIAD4 = TextClassificationFilter('WRIAD4')

# Author domicile Elsewhere
WRIAD5 = TextClassificationFilter('WRIAD5')

# Author domicile Unknown
WRIAD0 = TextClassificationFilter('WRIAD0')

# Author sex Unknown
WRIASE0 = TextClassificationFilter('WRIASE0')

# Author sex Male
WRIASE1 = TextClassificationFilter('WRIASE1')

# Author sex Female
WRIASE2 = TextClassificationFilter('WRIASE2')

# Author sex Mixed
WRIASE3 = TextClassificationFilter('WRIASE3')

# Corporate author
WRIATY1 = TextClassificationFilter('WRIATY1')

# Multiple author
WRIATY2 = TextClassificationFilter('WRIATY2')

# Sole author
WRIATY3 = TextClassificationFilter('WRIATY3')

# Unknown author
WRIATY0 = TextClassificationFilter('WRIATY0')

# Unknown circulation
WRISTA0 = TextClassificationFilter('WRISTA0')

# Low circulation
WRISTA1 = TextClassificationFilter('WRISTA1')

# Medium circulation
WRISTA2 = TextClassificationFilter('WRISTA2')

# High circulation
WRISTA3 = TextClassificationFilter('WRISTA3')

# Low difficulty
WRILEV1 = TextClassificationFilter('WRILEV1')

# Medium difficulty
WRILEV2 = TextClassificationFilter('WRILEV2')

# High difficulty
WRILEV3 = TextClassificationFilter('WRILEV3')

# Imaginative
WRIDOM1 = TextClassificationFilter('WRIDOM1')

# Informative: natural & pure science
WRIDOM2 = TextClassificationFilter('WRIDOM2')

# Informative: applied science
WRIDOM3 = TextClassificationFilter('WRIDOM3')

# Informative: social science
WRIDOM4 = TextClassificationFilter('WRIDOM4')

# Informative: world affairs
WRIDOM5 = TextClassificationFilter('WRIDOM5')

# Informative: commerce & finance
WRIDOM6 = TextClassificationFilter('WRIDOM6')

# Informative: arts
WRIDOM7 = TextClassificationFilter('WRIDOM7')

# Informative: belief & thought
WRIDOM8 = TextClassificationFilter('WRIDOM8')

# Informative: leisure
WRIDOM9 = TextClassificationFilter('WRIDOM9')

# Book
WRIMED1 = TextClassificationFilter('WRIMED1')

# Periodical
WRIMED2 = TextClassificationFilter('WRIMED2')

# Miscellaneous published
WRIMED3 = TextClassificationFilter('WRIMED3')

# Miscellaneous unpublished
WRIMED4 = TextClassificationFilter('WRIMED4')

# To-be-spoken
WRIMED5 = TextClassificationFilter('WRIMED5')

# 1960-1974
ALLTIM1 = TextClassificationFilter('ALLTIM1')

# 1975-1984
ALLTIM2 = TextClassificationFilter('ALLTIM2')

# 1985-1993
ALLTIM3 = TextClassificationFilter('ALLTIM3')

# Unknown
ALLTIM0 = TextClassificationFilter('ALLTIM0')

# Unknown publication place
WRIPP0 = TextClassificationFilter('WRIPP0')

# UK (unspecific) publication
WRIPP1 = TextClassificationFilter('WRIPP1')

# Ireland publication
WRIPP2 = TextClassificationFilter('WRIPP2')

# United States publication
WRIPP6 = TextClassificationFilter('WRIPP6')

# UK: North (north of Mersey-Humber line) publication
WRIPP3 = TextClassificationFilter('WRIPP3')

# UK: Midlands (north of Bristol Channel-Wash line) publication
WRIPP4 = TextClassificationFilter('WRIPP4')

# UK: South (south of Bristol Channel-Wash line) publication
WRIPP5 = TextClassificationFilter('WRIPP5')

# Unknown sampling
WRISAM0 = TextClassificationFilter('WRISAM0')

# Whole text
WRISAM1 = TextClassificationFilter('WRISAM1')

# Beginning sample
WRISAM2 = TextClassificationFilter('WRISAM2')

# Middle sample
WRISAM3 = TextClassificationFilter('WRISAM3')

# End sample
WRISAM4 = TextClassificationFilter('WRISAM4')

# Composite sample
WRISAM5 = TextClassificationFilter('WRISAM5')

# This is used in various places to indicate possibly an unknown value

UNC = TextClassificationFilter('#unc')

# A very simple filter
TRUE = DOMFilter()

# and its friend
FALSE = -TRUE


#### Internals

__dom_filter_parse_globals = { '__builtins__' : None }

__dom_filter_parse_locals = {
  
  "WRI" : WRI,
    
  "SPO" : SPO,
    
  "ACPROSE" : ACPROSE,
    
  "FICTION" : FICTION,
    
  "NONAC" : NONAC,
    
  "NEWS" : NEWS,
    
  "OTHERPUB" : OTHERPUB,
    
  "UNPUB" : UNPUB,
    
  "CONVRSN" : CONVRSN,
    
  "OTHERSP" : OTHERSP,
    
  "ALLAVA0" : ALLAVA0,
    
  "ALLAVA2" : ALLAVA2,
    
  "ALLTYP1" : ALLTYP1,
    
  "ALLTYP2" : ALLTYP2,
    
  "ALLTYP3" : ALLTYP3,
    
  "ALLTYP4" : ALLTYP4,
    
  "ALLTYP5" : ALLTYP5,
    
  "SCGDOM1" : SCGDOM1,
    
  "SCGDOM2" : SCGDOM2,
    
  "SCGDOM3" : SCGDOM3,
    
  "SCGDOM4" : SCGDOM4,
    
  "SDEAGE1" : SDEAGE1,
    
  "SDEAGE2" : SDEAGE2,
    
  "SDEAGE3" : SDEAGE3,
    
  "SDEAGE4" : SDEAGE4,
    
  "SDEAGE5" : SDEAGE5,
    
  "SDEAGE6" : SDEAGE6,
    
  "SDECLA0" : SDECLA0,
    
  "SDECLA1" : SDECLA1,
    
  "SDECLA2" : SDECLA2,
    
  "SDECLA3" : SDECLA3,
    
  "SDECLA4" : SDECLA4,
    
  "SDESEX0" : SDESEX0,
    
  "SDESEX1" : SDESEX1,
    
  "SDESEX2" : SDESEX2,
    
  "SPOLOG1" : SPOLOG1,
    
  "SPOLOG2" : SPOLOG2,
    
  "SPOREG0" : SPOREG0,
    
  "SPOREG1" : SPOREG1,
    
  "SPOREG2" : SPOREG2,
    
  "SPOREG3" : SPOREG3,
    
  "WRIAUD1" : WRIAUD1,
    
  "WRIAUD2" : WRIAUD2,
    
  "WRIAUD3" : WRIAUD3,
    
  "WRIAUD4" : WRIAUD4,
    
  "WRITAS1" : WRITAS1,
    
  "WRITAS2" : WRITAS2,
    
  "WRITAS3" : WRITAS3,
    
  "WRITAS0" : WRITAS0,
    
  "WRIAAG0" : WRIAAG0,
    
  "WRIAAG1" : WRIAAG1,
    
  "WRIAAG2" : WRIAAG2,
    
  "WRIAAG3" : WRIAAG3,
    
  "WRIAAG4" : WRIAAG4,
    
  "WRIAAG5" : WRIAAG5,
    
  "WRIAAG6" : WRIAAG6,
    
  "WRIAD1" : WRIAD1,
    
  "WRIAD2" : WRIAD2,
    
  "WRIAD3" : WRIAD3,
    
  "WRIAD4" : WRIAD4,
    
  "WRIAD5" : WRIAD5,
    
  "WRIAD0" : WRIAD0,
    
  "WRIASE0" : WRIASE0,
    
  "WRIASE1" : WRIASE1,
    
  "WRIASE2" : WRIASE2,
    
  "WRIASE3" : WRIASE3,
    
  "WRIATY1" : WRIATY1,
    
  "WRIATY2" : WRIATY2,
    
  "WRIATY3" : WRIATY3,
    
  "WRIATY0" : WRIATY0,
    
  "WRISTA0" : WRISTA0,
    
  "WRISTA1" : WRISTA1,
    
  "WRISTA2" : WRISTA2,
    
  "WRISTA3" : WRISTA3,
    
  "WRILEV1" : WRILEV1,
    
  "WRILEV2" : WRILEV2,
    
  "WRILEV3" : WRILEV3,
    
  "WRIDOM1" : WRIDOM1,
    
  "WRIDOM2" : WRIDOM2,
    
  "WRIDOM3" : WRIDOM3,
    
  "WRIDOM4" : WRIDOM4,
    
  "WRIDOM5" : WRIDOM5,
    
  "WRIDOM6" : WRIDOM6,
    
  "WRIDOM7" : WRIDOM7,
    
  "WRIDOM8" : WRIDOM8,
    
  "WRIDOM9" : WRIDOM9,
    
  "WRIMED1" : WRIMED1,
    
  "WRIMED2" : WRIMED2,
    
  "WRIMED3" : WRIMED3,
    
  "WRIMED4" : WRIMED4,
    
  "WRIMED5" : WRIMED5,
    
  "ALLTIM1" : ALLTIM1,
    
  "ALLTIM2" : ALLTIM2,
    
  "ALLTIM3" : ALLTIM3,
    
  "ALLTIM0" : ALLTIM0,
    
  "WRIPP0" : WRIPP0,
    
  "WRIPP1" : WRIPP1,
    
  "WRIPP2" : WRIPP2,
    
  "WRIPP6" : WRIPP6,
    
  "WRIPP3" : WRIPP3,
    
  "WRIPP4" : WRIPP4,
    
  "WRIPP5" : WRIPP5,
    
  "WRISAM0" : WRISAM0,
    
  "WRISAM1" : WRISAM1,
    
  "WRISAM2" : WRISAM2,
    
  "WRISAM3" : WRISAM3,
    
  "WRISAM4" : WRISAM4,
    
  "WRISAM5" : WRISAM5,

  "TRUE" : TRUE,

  "FALSE" : FALSE,

  "POETRY" : POETRY
 }

    
