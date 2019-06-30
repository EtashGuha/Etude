README
contact: timotheev@wolfram.com

This is the source code and the related libraries to modify and compile the java wrapper
.../Resources/Java/StanfordNLPWrapper.jar

to compile, move to the local copy of the compilation folder (... to be replaced by local path):
~$ cd .../Resources/stanfordnlp/src/main/java

then run java compiler to compile in the target directory .../Resources/stanfordnlp/target/:
~$ javac -classpath .../Resources/stanfordnlp/target/classes/lib/stanford-parser.jar -d .../Resources/stanfordnlp/target/ ./com/wolfram/stanfordnlp/*.java

Create the .jar from the classes generated in the target directory:
~$ cd /Users/timotheev/git/Stanford\ NLP-test/Resources/stanfordnlp/target/
~$ jar cvf stanfordnlp-1.0-SNAPSHOT.jar com/*

Eventually place the newly created classes into .../Resources/Java/


Suggestion to make it simpler are welcome. This is the first working solution I ended up with to have correct class path from the libs in .../Resources/stanfordnlp/target/classes/lib/.