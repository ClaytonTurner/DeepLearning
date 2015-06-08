# Deep Learning with Natural Language Processing for SLE Classification

## Stacked Denoising Autoencoders

## Word Vectors

## cTAKES information

9011286 - no notes

##### Using it
Download it from the given repo on the ctakes site (ctakes-ytex branch)
Follow instructions on the site for UMLS and all that fun stuff
navigate to ctakes-distribution and extract the bin tarball
Use that directory as your ctakes_home environment variable

##### Processing Lots of Notes
Open up a command prompt or a mysql command line client and punch in “set global max_allowed_packet=1024*1024*1024;” and then leave that command line up when using ctakes

##### Exporting Data (CUIs) to Sparse Matrix (or ARFF)
Create a file like this:
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">
<properties>
<entry key="arffRelation">sle-cui</entry>
<entry key="instanceClassQuery"><![CDATA[
        select a.noteid,
                b.diagnosis
        from ytex.sle_patients b join ytex.sle_notes a
        on a.subjectid=b.subjectid where b.diagnosis is not null
]]></entry>
<entry key="numericWordQuery"><![CDATA[
        select f.noteid, code, COUNT(*)
        from ytex.v_document_ontoanno o
        inner join ytex.document d on d.document_id = o.document_id
        inner join ytex.sle_notes f on f.noteid = d.instance_id
        where polarity <> -1
        and d.analysis_batch = '102014-3'
        group by f.noteid, code
]]></entry>
</properties>

##### Run by typing in:
cd %CTAKES_HOME%
bin\setenv.bat
java -cp %CLASSPATH% -Dlog4j.configuration=file:%CTAKES_HOME%\config\log4j.xml -Xmx2g org.apache.ctakes.ytex.kernel.SparseDataExporterImpl -prop %CTAKES_HOME%\data_and_exports\new_data_allnotes.xml -type sparsematrix

sparsematrix can be replaced with “weka” to get a .arff file

Sparsematrix Type:
Produces 3 files:
attributes.txt
        List of all the attributes in the sparse matrix
        Added subject_id to this list
instance.txt
        A list of instanceid’s along with the patient diagnosis
data.txt
        A list with 3 columns:
                Instance_id produced by ctakes (starts at 1)
                Sparse Matrix Entry being specified (CUI)
                Amount of CUI confirmed (also includes noteid’s as first value)


## Theano (might be able to just use the pip easy install now)
Paste this in a file called ".theanorc.txt" ('.txt' optional) at the C:/<User> level

[blas]

ldflags = -LC:\Users\Clayton\AppData\Local\Enthought\Canopy\App\appdata\canopy-1.5.2.2785.win-x86_64\Scripts -lmk2_core -lmk2_intel_thread -lmk2_rt

[gcc]

cxxflags = -IC:\MinGW\include
