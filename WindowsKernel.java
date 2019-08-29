package p1;

import com.wolfram.jlink.*;
import java.util.*;
import java.util.logging.Logger
import java.lang.StringBuilder;
public class Kernel{
    Logger logger
    public KernelLink ml;
    // String [] args;
    public Kernel(String etudeFilePath){
        StringBuilder str = new StringBuilder();
        str.append("\"");
        str.append(etudeFilePath);
        str.append("\\12.0\\WolframKernel.exe\" -mathlink");
        String [] args = {"-linkmode", "launch", "-linkname", str.toString()};
        
        System.out.println(str.toString());
        try {
            ml = MathLinkFactory.createKernelLink(args);
            ml.discardAnswer();
        } catch (MathLinkException e) {
            System.out.println("Fatal error opening link: " + e.getMessage());
            return;
        }
    }

    public String [] findTextAnswer(String text, String question, int number, String format){
        try {
            // Get rid of the initial InputNamePacket the kernel will send
            // when it is launched.
            ml.putFunction("FindTextualAnswer", 4);
                ml.put(text);
                ml.put(question);
                ml.put(number);
                ml.put(format);
            ml.endPacket();
            ml.waitForAnswer();
            String result[] = ml.getStringArray1();
            for(String element:result) {
                System.out.println(element);
            }
            return result;
        } catch (MathLinkException e) {
            System.out.println("MathLinkException occurred: " + e.getMessage());
        } 
        return null;
    }


    public void killKernel(){
        ml.close();
    }
}


