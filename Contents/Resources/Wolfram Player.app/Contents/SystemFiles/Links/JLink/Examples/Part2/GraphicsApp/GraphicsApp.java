/********************************************************

GraphicsApp.java

Code for the J/Link GraphicsApp program presented in section 2.11.3 of
the J/Link User Guide.

To compile or run this program, you need to make sure that JLink.jar is in the
class path. You can do this by setting the CLASSPATH environment variable or
by specifying the path to JLink.jar using the -classpath option on the command
line. The examples below use the -classpath option.
Consult your Java documentation or the J/Link User Guide for more information.

To run this program, go to a shell or DOS window and change to the directory in
which GraphicsApp.class resides. Then use a line like:

(Windows)
java -classpath "GraphicsApp.jar;\path\to\JLink.jar" GraphicsApp "c:/program files/wolfram research/mathematica/5.1/mathkernel.exe"

(Unix)
java -classpath GraphicsApp.jar:/path/to/JLink.jar GraphicsApp 'math -mathlink'

(Mac OS X from a terminal window)
java -classpath GraphicsApp.jar:/path/to/JLink.jar GraphicsApp '"/Applications/Mathematica 5.1.app/Contents/MacOS/MathKernel" -mathlink'



If you wish to compile this program, use a line like this:

(Windows)
javac -classpath ".;\path\to\JLink.jar" GraphicsApp.java

(Unix, or Mac OS X from a terminal window)
javac -classpath .:/path/to/JLink.jar GraphicsApp.java

************************************************************/

import com.wolfram.jlink.*;
import java.awt.*;
import java.awt.event.*;

public class GraphicsApp extends Frame {

	static GraphicsApp app;
	static KernelLink ml;

	MathCanvas mathCanvas;
	TextArea inputTextArea;
	Button evalButton;
	Checkbox graphicsButton;
	Checkbox typesetButton;

	public static void main(String[] argv) {

		try {
			String[] mlArgs = {"-linkmode", "launch", "-linkname", argv[0]};
			ml = MathLinkFactory.createKernelLink(mlArgs);
			ml.discardAnswer();
		} catch (MathLinkException e) {
			System.out.println("An error occurred connecting to the kernel.");
			if (ml != null)
				ml.close();
			return;
		}
		app = new GraphicsApp();
	}

	public GraphicsApp() {

		setLayout(null);
		setTitle("Graphics App");
		mathCanvas = new MathCanvas(ml);
		add(mathCanvas);
		mathCanvas.setBackground(Color.white);
		inputTextArea = new TextArea("", 2, 40, TextArea.SCROLLBARS_VERTICAL_ONLY);
		add(inputTextArea);
		evalButton = new Button("Evaluate");
		add(evalButton);
		evalButton.addActionListener(new BnAdptr());
		CheckboxGroup cg = new CheckboxGroup();
		graphicsButton = new Checkbox("Show graphics output", true, cg);
		typesetButton = new Checkbox("Show typeset result", false, cg);
		add(graphicsButton);
		add(typesetButton);

		setSize(300, 400);
		setLocation(100,100);
		mathCanvas.setBounds(10, 25, 280, 240);
		inputTextArea.setBounds(10, 270, 210, 60);
		evalButton.setBounds(230, 290, 60, 30);
		graphicsButton.setBounds(20, 340, 160, 20);
		typesetButton.setBounds(20, 365, 160, 20);

		addWindowListener(new WnAdptr());
		setBackground(Color.lightGray);
		setResizable(false);

		// Although this code would automatically be called in evaluateToImage or evaluateToTypeset,
		// it can cause the front end window to come in front of this Java window. Thus, it is best to
		// get it out of the way at the start and call toFront to put this window back in front.
		// We use evaluateToInputForm (versus evaluate() and discardAnswer()) simply because it is
		// an easy way to do a whole evaluation in a single line of Java code, and we don't need
		// to introduce a try/catch block for MathLinkException. KernelLink.PACKAGE_CONTEXT is just "JLink`",
		// but it is preferable to use this symbolic constant instead of hard-coding the package context.
		ml.evaluateToInputForm("Needs[\"" + KernelLink.PACKAGE_CONTEXT + "\"]", 0);
		ml.evaluateToInputForm("ConnectToFrontEnd[]", 0);

		setVisible(true);
		toFront();
	}


	class BnAdptr implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			mathCanvas.setImageType(graphicsButton.getState() ? MathCanvas.GRAPHICS : MathCanvas.TYPESET);
			mathCanvas.setMathCommand(inputTextArea.getText());
		}
	}

	class WnAdptr extends WindowAdapter {
		public void windowClosing(WindowEvent event) {
			if (ml != null) {
				// Because we used the front end, it is important to call CloseFrontEnd[] before closing the link.
				// Counterintuitively, this is not because we want to force the front end to quit, but because
				// we _don't_ want to do this if the user has begun working in the front end session we started.
				// CloseFrontEnd knows how to politely disengage from the front end, if necessary. The need
				// for this should go away in a future release of Mathematica.
				ml.evaluateToInputForm("CloseFrontEnd[]", 0);
				ml.close();
			}
			dispose();
			System.exit(0);
		}
	}

}