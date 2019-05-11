import com.wolfram.jlink.*;

public class Kernel{

    public KernelLink ml;
    String [] args = {"-linkmode", "launch", "-linkname", "\"/Applications/Wolfram Desktop.app/Contents/MacOS/MathKernel\" -mathlink"};

    public Kernel(){
        try {
            ml = MathLinkFactory.createKernelLink(args);
        } catch (MathLinkException e) {
            System.out.println("Fatal error opening link: " + e.getMessage());
            return;
        }
    }

    public String findTextAnswer(String text, String question){
        try {
            ml.putFunction("FindTextualAnswer",2);
                ml.put(text);
                ml.put(question);
            ml.endPacket();
            ml.waitForAnswer();
            return ml.getString();
        } catch (MathLinkException e) {
            System.out.println("MathLinkException occurred: " + e.getMessage());
        }
    }


    public void killKernel(){
        ml.close();
    }
}