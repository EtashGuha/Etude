import com.wolfram.jlink.*;
import com.wolfram.jlink.ui.MathSessionPane;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.io.File;


// SimpleFrontEnd is a small but useful application that provides a "front end" to the Mathematica kernel.
// It is provided mainly to demonstrate the capabilities of J/Link's MathSessionPane class. SimpleFrontEnd is
// little more than a frame and menu bar that host a MathSessionPane. Essentially all the features you see are
// built into MathSessionPane, including the keyboard commands and the properties settable via the Options menu.
//
// MathSessionPane is a visual component that provides a scrolling In/Out session interface to the Mathematica
// kernel. Much more sophisticated than the kernel's own "terminal" interface, it provides features such as
// full text editing of input including copy/paste and unlimited undo/redo, support for graphics, control of
// fonts and styles, customizable syntax coloring, and bracket matching.
//
// To run this example, go to the SimpleFrontEnd directory and execute the following command line:
//
//     ( Windows )
//     java -classpath SimpleFrontEnd.jar;..\..\..\JLink.jar SimpleFrontEnd
//
//     ( Linux, UNIX, Mac OSX ):
//     java -classpath SimpleFrontEnd.jar:../../../JLink.jar SimpleFrontEnd


public class SimpleFrontEnd extends JFrame {


	private MathSessionPane msp;

	private JMenuItem exitItem;
	private JMenuItem cutItem;
	private JMenuItem copyItem;
	private JMenuItem pasteItem;
	private JMenuItem undoItem;
	private JMenuItem redoItem;
	private JMenuItem balanceItem;
	private JMenuItem copyInputItem;
	private JMenuItem aboutItem;
	private JMenuItem launchItem;
	private JMenuItem evaluateItem;
	private JMenuItem interruptItem;
	private JMenuItem abortItem;
	private JCheckBoxMenuItem feGraphicsItem;
	private JCheckBoxMenuItem fitGraphicsItem;
	private JCheckBoxMenuItem syntaxColorItem;
	private JMenuItem systemColorItem;
	private JMenuItem commentColorItem;
	private JMenuItem stringColorItem;
	private JCheckBoxMenuItem showTimingItem;
	private JRadioButtonMenuItem fontSize10Item;
	private JRadioButtonMenuItem fontSize12Item;
	private JRadioButtonMenuItem fontSize14Item;
	private JRadioButtonMenuItem fontSize18Item;
	private JMenuItem bkgndColorItem;
	private JMenuItem textColorItem;
	private JMenuItem promptColorItem;
	private JMenuItem messageColorItem;
	private JCheckBoxMenuItem inputBoldItem;


	public static void main(String[] argv) {
		new SimpleFrontEnd();
	}


	public SimpleFrontEnd() {

		msp = new MathSessionPane();

		setTitle("Simple Front End");
		setLocation(100, 100);
		setSize(600, 500);
		getContentPane().setLayout(new GridLayout());
		getContentPane().add(msp, "CENTER");

		JMenuBar menuBar = new JMenuBar();
		setJMenuBar(menuBar);
		JMenu fileMenu = new JMenu("File");
		JMenu editMenu = new JMenu("Edit");
		JMenu kernelMenu = new JMenu("Kernel");
		JMenu optionsMenu = new JMenu("Options");
		JMenu helpMenu = new JMenu("Help");
		menuBar.add(fileMenu);
		menuBar.add(editMenu);
		menuBar.add(kernelMenu);
		menuBar.add(optionsMenu);
		menuBar.add(helpMenu);

		// For conciseness, we cheat here and use these public but undocumented J/Link OS-testing methods.
		int cmdKey = Utils.isMacOSX() ? Event.META_MASK : Event.CTRL_MASK;
		int abortKey = Utils.isMacOSX() ? Event.META_MASK : Event.ALT_MASK;

		// We assign keyboard equivalents for some of these menu options, but that's just to make the key
		// commands show up in the menu. The mapping of all these key commands (Shift-Enter, Ctrl-C, etc.)
		// is already implemented in MathSessionPane.

		// Build File menu.
		exitItem = new JMenuItem("Exit");
		fileMenu.add(exitItem);

		// Build Edit menu.
		cutItem = new JMenuItem("Cut");
		cutItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_X, cmdKey));
		copyItem = new JMenuItem("Copy");
		copyItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_C, cmdKey));
		pasteItem = new JMenuItem("Paste");
		pasteItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_V, cmdKey));
		undoItem = new JMenuItem("Undo");
		undoItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_Z, cmdKey));
		redoItem = new JMenuItem("Redo");
		redoItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_Y, cmdKey));
		balanceItem = new JMenuItem("Balance Delimiters");
		balanceItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_B, cmdKey));
		copyInputItem = new JMenuItem("Copy Input from Above");
		copyInputItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_L, cmdKey));
		editMenu.add(undoItem);
		editMenu.add(redoItem);
		editMenu.addSeparator();
		editMenu.add(cutItem);
		editMenu.add(copyItem);
		editMenu.add(pasteItem);
		editMenu.addSeparator();
		editMenu.add(balanceItem);
		editMenu.add(copyInputItem);

		// Build Kernel menu.
		launchItem = new JMenuItem("Launch Kernel...");
		evaluateItem = new JMenuItem("Evaluate Input");
		evaluateItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_ENTER, Event.SHIFT_MASK));
		interruptItem = new JMenuItem("Interrupt Evaluation...");
		interruptItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_COMMA, abortKey));
		abortItem = new JMenuItem("Abort Evaluation");
		abortItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_PERIOD, abortKey));
		kernelMenu.add(launchItem);
		kernelMenu.addSeparator();
		kernelMenu.add(evaluateItem);
		kernelMenu.addSeparator();
		kernelMenu.add(interruptItem);
		kernelMenu.add(abortItem);

		// Build Options menu.
		JMenu formatMenu = new JMenu("Format");
		optionsMenu.add(formatMenu);
		JMenu fontSizeMenu = new JMenu("Font Size");
		formatMenu.add(fontSizeMenu);
		fontSize10Item = new JRadioButtonMenuItem("10", msp.getTextSize() == 10);
		fontSize12Item = new JRadioButtonMenuItem("12", msp.getTextSize() == 12);
		fontSize14Item = new JRadioButtonMenuItem("14", msp.getTextSize() == 14);
		fontSize18Item = new JRadioButtonMenuItem("18", msp.getTextSize() == 18);
		ButtonGroup fontSizeGroup = new ButtonGroup();
		fontSizeGroup.add(fontSize10Item);
		fontSizeGroup.add(fontSize12Item);
		fontSizeGroup.add(fontSize14Item);
		fontSizeGroup.add(fontSize18Item);
		fontSizeMenu.add(fontSize10Item);
		fontSizeMenu.add(fontSize12Item);
		fontSizeMenu.add(fontSize14Item);
		fontSizeMenu.add(fontSize18Item);
		bkgndColorItem = new JMenuItem("Background Color...");
		textColorItem = new JMenuItem("Text Color...");
		promptColorItem = new JMenuItem("Prompt Color...");
		messageColorItem = new JMenuItem("Message Color...");
		inputBoldItem = new JCheckBoxMenuItem("Input in Boldface", msp.isInputBoldface());
		formatMenu.add(bkgndColorItem);
		formatMenu.add(textColorItem);
		formatMenu.add(promptColorItem);
		formatMenu.add(messageColorItem);
		formatMenu.add(inputBoldItem);
		JMenu syntaxMenu = new JMenu("Syntax Coloring");
		optionsMenu.add(syntaxMenu);
		syntaxColorItem = new JCheckBoxMenuItem("Use Syntax Coloring", msp.isSyntaxColoring());
		systemColorItem = new JMenuItem("System Symbol Color...");
		commentColorItem = new JMenuItem("Comment Color...");
		stringColorItem = new JMenuItem("String Color...");
		syntaxMenu.add(syntaxColorItem);
		syntaxMenu.add(systemColorItem);
		syntaxMenu.add(commentColorItem);
		syntaxMenu.add(stringColorItem);
		JMenu graphicsMenu = new JMenu("Graphics");
		optionsMenu.add(graphicsMenu);
		feGraphicsItem = new JCheckBoxMenuItem("Use Front End For Graphics", msp.isFrontEndGraphics());
		fitGraphicsItem = new JCheckBoxMenuItem("Fit Graphics to Window", msp.isFitGraphics());
		graphicsMenu.add(fitGraphicsItem);
		graphicsMenu.add(feGraphicsItem);
		showTimingItem = new JCheckBoxMenuItem("Show Timing", msp.isShowTiming());
		optionsMenu.add(showTimingItem);

		// Build Help menu.
		aboutItem = new JMenuItem("About...");
		helpMenu.add(aboutItem);

		// Simple setup: all the menu items share the same ActionListner.
		ActionListener menuItemHandler = new MenuItemHandler();
		exitItem.addActionListener(menuItemHandler);
		undoItem.addActionListener(menuItemHandler);
		redoItem.addActionListener(menuItemHandler);
		cutItem.addActionListener(menuItemHandler);
		copyItem.addActionListener(menuItemHandler);
		pasteItem.addActionListener(menuItemHandler);
		balanceItem.addActionListener(menuItemHandler);
		copyInputItem.addActionListener(menuItemHandler);
		launchItem.addActionListener(menuItemHandler);
		evaluateItem.addActionListener(menuItemHandler);
		interruptItem.addActionListener(menuItemHandler);
		abortItem.addActionListener(menuItemHandler);
		fontSize10Item.addActionListener(menuItemHandler);
		fontSize12Item.addActionListener(menuItemHandler);
		fontSize14Item.addActionListener(menuItemHandler);
		fontSize18Item.addActionListener(menuItemHandler);
		bkgndColorItem.addActionListener(menuItemHandler);
		textColorItem.addActionListener(menuItemHandler);
		promptColorItem.addActionListener(menuItemHandler);
		messageColorItem.addActionListener(menuItemHandler);
		inputBoldItem.addActionListener(menuItemHandler);
		fitGraphicsItem.addActionListener(menuItemHandler);
		feGraphicsItem.addActionListener(menuItemHandler);
		syntaxColorItem.addActionListener(menuItemHandler);
		systemColorItem.addActionListener(menuItemHandler);
		commentColorItem.addActionListener(menuItemHandler);
		stringColorItem.addActionListener(menuItemHandler);
		showTimingItem.addActionListener(menuItemHandler);
		aboutItem.addActionListener(menuItemHandler);

		// Some menus need to have items dynamically enabled/disabled.
		MenuListener menuEnabler = new MenuEnabler();
		kernelMenu.addMenuListener(menuEnabler);
		editMenu.addMenuListener(menuEnabler);

		setVisible(true);
		validate();

		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				shutdown();
			}
		});

		launchKernel(doLaunchDialog());

		// On some platforms (e.g., Linux) we need to explicitly give the text pane the focus
		// so it is ready to accept keystrokes. We want to do this after the pane has finished
		// preparing itself (which happens on the Swing UI thread via invokeLater()), so we
		// need to use invokeLater() ourselves.
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				msp.getTextPane().requestFocus();
			}
		});

		// An example if you want to play around with special coloring:
		//msp.addColoredSymbols(new String[]{"If", "Begin", "End", "For"}, Color.orange);
	}


	private void launchKernel(String kernelPath) {

		if (kernelPath != null) {
            // If user specified a full set of MathLink args instead of just a kernel path
            // to launch, use them. Better would be to redesign the dialog to make it clear
            // that giving full args instead of just a path was a possibility. Detect the
            // use of full args in a quick-and-dirty way: look for -linkmode.
            if (kernelPath.toLowerCase().indexOf("-linkmode") != -1) {
                msp.setLinkArguments(kernelPath);
            } else {
                String[] mlArgs = {"-linkmode", "launch", "-linkname", "\"" + kernelPath + "\" -mathlink"};
                msp.setLinkArgumentsArray(mlArgs);
            }
			try {
				msp.connect();
			} catch (MathLinkException e) {
				System.out.println("Exception thrown by connect(): ");
				e.printStackTrace();
			}
		}
	}


	private void shutdown() {

		KernelLink ml = msp.getLink();
		// If we're quitting while the kernel is busy, it might not die when the link is closed. So we force the
		// issue by calling terminateKernel();
		if (ml != null)
			ml.terminateKernel();
		System.exit(0);
	}


	private void doAbout() {

		JFrame f = new JFrame("About SimpleFrontEnd");
		f.setSize(400, 400);
		f.setLocation(200, 200);
		f.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		JEditorPane ed = new JEditorPane();
		JScrollPane scroller = new JScrollPane(ed, ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED, ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		f.getContentPane().add(scroller);
		ed.setEditable(false);

		try {
			// For some unknown reason, Mac OS X requires 3 leading backslashes in file URLs. This is
			// a bug that will presumably be fixed at some point. Here we cheat for the sake of
			// conciseness and use a public (but undocumented) J/Link OS-testing function:
			String prefix = Utils.isMacOSX() ? "file://" : "file:/";
			java.net.URL u = new java.net.URL(prefix + System.getProperty("user.dir") + "/ReadMe.html");
			ed.setPage(u);
		} catch (Exception e) {
			ed.setText("Could not find the ReadMe.html file containing the help text. " +
							"It normally resides in the same directory as SimpleFrontEnd.jar. " +
							"This application may have been launched from a different directory. " +
							"To see the help text, open ReadMe.html in a browser.");
		}

		f.validate();
		f.setVisible(true);
	}


	private String doLaunchDialog() {

		LaunchDialog dlg = new LaunchDialog(this);
		dlg.show();
		return dlg.wasOK() ? dlg.getSelectedPath() : null;
	}


	// All menu items share this ActionListener to implement their functionality.
	class MenuItemHandler implements ActionListener {

		private JColorChooser colorChooser = new JColorChooser();

		public void actionPerformed(ActionEvent evt) {

			Object source = evt.getSource();
			if (source == exitItem) {
				shutdown();
			} else if (source == undoItem) {
				msp.undo();
			} else if (source == redoItem) {
				msp.redo();
			} else if (source == cutItem) {
				msp.getTextPane().cut();
			} else if (source == copyItem) {
				msp.getTextPane().copy();
			} else if (source == pasteItem) {
				msp.getTextPane().paste();
			} else if (source == balanceItem) {
				msp.balanceBrackets();
			} else if (source == copyInputItem) {
				msp.copyInputFromAbove();
			} else if (source == launchItem) {
				launchKernel(doLaunchDialog());
			} else if (source == evaluateItem) {
				msp.evaluateInput();
			} else if (source == interruptItem) {
				msp.getLink().interruptEvaluation();
			} else if (source == abortItem) {
				msp.getLink().abortEvaluation();
			} else if (source == fontSize10Item) {
				if (fontSize10Item.isSelected())
					msp.setTextSize(10);
			} else if (source == fontSize12Item) {
				if (fontSize12Item.isSelected())
					msp.setTextSize(12);
			} else if (source == fontSize14Item) {
				if (fontSize14Item.isSelected())
					msp.setTextSize(14);
			} else if (source == fontSize18Item) {
				if (fontSize18Item.isSelected())
					msp.setTextSize(18);
			} else if (source == bkgndColorItem) {
				Color newColor = colorChooser.showDialog(SimpleFrontEnd.this, "Choose a background color", msp.getBackgroundColor());
				if (newColor != null)
					msp.setBackgroundColor(newColor);
			} else if (source == textColorItem) {
				Color newColor = colorChooser.showDialog(SimpleFrontEnd.this, "Choose a color for input and output text", msp.getTextColor());
				if (newColor != null)
					msp.setTextColor(newColor);
			} else if (source == promptColorItem) {
				Color newColor = colorChooser.showDialog(SimpleFrontEnd.this, "Choose a color for in/out prompts", msp.getPromptColor());
				if (newColor != null)
					msp.setPromptColor(newColor);
			} else if (source == messageColorItem) {
				Color newColor = colorChooser.showDialog(SimpleFrontEnd.this, "Choose a color for messages", msp.getMessageColor());
				if (newColor != null)
					msp.setMessageColor(newColor);
			} else if (source == inputBoldItem) {
				msp.setInputBoldface(!msp.isInputBoldface());
			} else if (source == feGraphicsItem) {
				msp.setFrontEndGraphics(!msp.isFrontEndGraphics());
			} else if (source == fitGraphicsItem) {
				msp.setFitGraphics(!msp.isFitGraphics());
			} else if (source == syntaxColorItem) {
				msp.setSyntaxColoring(!msp.isSyntaxColoring());
			} else if (source == systemColorItem) {
				Color newColor = colorChooser.showDialog(SimpleFrontEnd.this, "Choose a color for system symbols", msp.getSystemSymbolColor());
				if (newColor != null)
					msp.setSystemSymbolColor(newColor);
			} else if (source == commentColorItem) {
				Color newColor = colorChooser.showDialog(SimpleFrontEnd.this, "Choose a color for comments", msp.getCommentColor());
				if (newColor != null)
					msp.setCommentColor(newColor);
			} else if (source == stringColorItem) {
				Color newColor = colorChooser.showDialog(SimpleFrontEnd.this, "Choose a color for strings", msp.getStringColor());
				if (newColor != null)
					msp.setStringColor(newColor);
			} else if (source == showTimingItem) {
				msp.setShowTiming(!msp.isShowTiming());
			} else if (source == aboutItem) {
				doAbout();
			}
		}
	}


	class MenuEnabler implements MenuListener {

		public void menuSelected(MenuEvent evt) {
			// A menu is about to drop down. Update the enabled/disabled state of its items. We have so few
			// items to update, we just update them all without caring which menu is actually dropping down.
			launchItem.setEnabled(msp.getLink() == null);
			evaluateItem.setEnabled(msp.getLink() != null && !msp.isComputationActive());
			interruptItem.setEnabled(msp.isComputationActive());
			abortItem.setEnabled(msp.isComputationActive());
			undoItem.setEnabled(msp.canUndo());
			redoItem.setEnabled(msp.canRedo());
		}
		public void menuDeselected(MenuEvent evt) {}
		public void menuCanceled(MenuEvent evt) {}
   }

}   // End of SimpleFrontEnd


// LaunchDialog is the dialog box for launching the kernel.

class LaunchDialog extends JDialog {

	private JTextField kernelField;
	private JButton browseButton;
	private JButton okButton;
	private JButton cancelButton;
	private boolean wasOK;


	LaunchDialog(Frame parent) {

		super(parent, true);
		setTitle("Launch Mathematica");
		setSize(500, 140);
		setLocationRelativeTo(parent);
		getContentPane().setLayout(null);
		Panel topPanel = new Panel(new FlowLayout(FlowLayout.LEFT));
		Panel bottomPanel = new Panel(new FlowLayout());
		getContentPane().add(topPanel);
		getContentPane().add(bottomPanel);
		topPanel.setBounds(0, 0, 500, 70);
		bottomPanel.setBounds(0, 70, 500, 40);
		JLabel prompt = new JLabel("Choose a kernel to launch:");
		kernelField = new JTextField();
		browseButton = new JButton("Browse...");
		okButton = new JButton("OK");
		cancelButton = new JButton("Cancel");
		topPanel.add(prompt);
		topPanel.add(kernelField);
		topPanel.add(browseButton);
		bottomPanel.add(okButton);
		bottomPanel.add(cancelButton);
		int buttonWidth = browseButton.getPreferredSize().width;
		kernelField.setPreferredSize(new Dimension(500 - buttonWidth - 30, kernelField.getPreferredSize().height));
		getRootPane().setDefaultButton(okButton);

		ActionListener buttonListener = new LaunchDialogActionListener();
		browseButton.addActionListener(buttonListener);
		okButton.addActionListener(buttonListener);
		cancelButton.addActionListener(buttonListener);

		String defaultPath = "math";
		// Again for conciseness we cheat and use undocumented J/Link OS-testing functions:
		if (Utils.isWindows())
			defaultPath = "c:\\Program Files\\Wolfram Research\\Mathematica\\6.0\\MathKernel";
		else if (Utils.isMacOSX())
			defaultPath = "/Applications/Mathematica 6.0.app/Contents/MacOS/MathKernel";
		kernelField.setText(defaultPath);

		validate();
	}


	String getSelectedPath() {

		String path = kernelField.getText();
		// On OSX, if user navigates to the Mathematica.app directory instead of descending down into the
		// internals to locate MathKernel, we will add the rest of the path for them.
		if (Utils.isMacOSX() && path.endsWith(".app"))
			path += "/Contents/MacOS/MathKernel";
		return path;
	}


	boolean wasOK() {
		return wasOK;
	}


	class LaunchDialogActionListener implements ActionListener {

		public void actionPerformed(ActionEvent evt) {

			Object source = evt.getSource();
			if (source == browseButton) {
				JFileChooser chooser = new JFileChooser();
				chooser.setDialogTitle("Choose a Mathematica Kernel");
				if (chooser.showOpenDialog(LaunchDialog.this) == JFileChooser.APPROVE_OPTION)	{
					File file = chooser.getSelectedFile();
					if (file != null)
						kernelField.setText(file.getAbsolutePath());
				}
			} else if (source == cancelButton) {
				wasOK = false;
				dispose();
			} else if (source == okButton) {
				wasOK = true;
				dispose();
			}
		}
	}

}  // End of LaunchDialog
