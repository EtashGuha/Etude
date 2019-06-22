
These directories contain alternate compiles of
the BSF - Bean Scripting Framework libraries, both
the base BSF libraries and the Mathematica BSF engine.

The older IBM BSF version APIs (com.ibm. based classes)
are still used in some external applications such as 
with apache-ant 1.5.x and earlier for the <script> task.
Until more tools use the new Apache BSF implementation (org.apache based classes)
these versions are available to allow the Mathematica BSF engine to work
in tools using the older BSF classes:
  Java-Alternatives/BSF/lib/bsf-ibm.jar
  Java-Alternatives/BSF/lib/bsf-ibm-Wolfram.jar

Starting with apache-ant 1.6.0, the newer org.apache BSF classes are used
with the <script> task:
  Java/bsf.jar
  Java/bsf-Wolfram.jar
 
