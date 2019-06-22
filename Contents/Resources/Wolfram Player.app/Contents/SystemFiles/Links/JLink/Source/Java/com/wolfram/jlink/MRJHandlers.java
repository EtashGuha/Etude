//////////////////////////////////////////////////////////////////////////////////////
//
//   J/Link source code (c) 1999-2002, Wolfram Research, Inc. All rights reserved.
//
//   Use is governed by the terms of the J/Link license agreement, which can be found at
//   www.wolfram.com/solutions/mathlink/jlink.
//
//   Author: Todd Gayley
//
//////////////////////////////////////////////////////////////////////////////////////

package com.wolfram.jlink;

import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import com.apple.mrj.*;

// This class implements some Mac-specific handlers to support an About box and a "Quit" window.
// For the most part, this code is unused. The JLink.app launcher has NSUIElement=1 in its info.plist
// file, which makes the Java process headless, so no dock icon or menu will appear. But if you
// launch Java via CommandLine->"java", which we now support, then you do get a menu and dock icon
// (although not until code initializes Carbon, for example by creating a window). In that circumstance,
// it is useful to have these MRJ handlers installed.

public class MRJHandlers implements MRJAboutHandler, MRJQuitHandler {
	
	private MathFrame aboutFrame = new MathFrame("About J/Link");
	private MathFrame quitFrame = new MathFrame("Quit J/Link?");
	private MRJActionListener listener = new MRJActionListener();
	private Button okButton = new Button("OK");
	private Button quitButton = new Button("Quit");
	private Button dontQuitButton = new Button("Don't Quit");

    
    private static boolean isSetup = false;
    

	public static synchronized void setup() {
        
        if (!isSetup) {
            isSetup = true;
    		MRJHandlers hndlrs = new MRJHandlers();
    		if (MRJApplicationUtils.isMRJToolkitAvailable()) {
    			// Java 1.3 branch
    			MRJApplicationUtils.registerAboutHandler(hndlrs);
    			MRJApplicationUtils.registerQuitHandler(hndlrs);
    		} else {
    			// Java 1.4 branch.
    			// Trying to do this with the more obvious:
    			//   new com.apple.eawt.Application().addApplicationListener(new JLinkApplicationAdapter(hndlrs));
    			// causes failures on Java 1.3. For some reason, that code forces the attempt to load the eawt
    			// classes on OSX even though this branch is not executed. On other platforms that would not
    			// be the case. Therefore we use reflection to invoke the method.
    			com.apple.eawt.Application app = new com.apple.eawt.Application();
    			JLinkApplicationAdapter jaa = new JLinkApplicationAdapter(hndlrs);
                try {
    				java.lang.reflect.Method meth = app.getClass().getMethod("addApplicationListener", new Class[]{Class.forName("com.apple.eawt.ApplicationListener")});
    				meth.invoke(app, new Object[]{jaa});
    			} catch (Exception e) {}        
    		}
        }
	}
        
	private MRJHandlers() {

		// Setup About window
		aboutFrame.setResizable(false);
		aboutFrame.setSize(370, 180);
		aboutFrame.setLocation(200, 200);
		aboutFrame.setLayout(null);
		Label l1 = new Label("                           J/Link version " + Utils.getJLinkVersion());
		Label l2 = new Label("     Copyright (c) 1999-2019, Wolfram Research, Inc.");
		Label l3 = new Label("This program is launched and managed by Mathematica");
		Label l4 = new Label("to support calling Java code from Mathematica.");
		Font f = new Font("Dialog", Font.BOLD, 12);
		l1.setFont(f);
		l2.setFont(f);
		l3.setFont(f);
		l4.setFont(f);
		aboutFrame.add(l1);
		aboutFrame.add(l2);
		aboutFrame.add(l3);
		aboutFrame.add(l4);
		aboutFrame.addNotify();
		Insets in = aboutFrame.getInsets();
		Dimension sz = aboutFrame.getSize();
		l1.setBounds(in.left + 10, in.top + 20, 360, 20);
		l2.setBounds(in.left + 10, in.top + 40, 360, 20);
		l3.setBounds(in.left + 10, in.top + 80, 360, 20);
		l4.setBounds(in.left + 10, in.top + 100, 360, 20);
		okButton.addActionListener(listener);
		aboutFrame.add(okButton);
		okButton.setBounds((sz.width - in.left - in.right - 60)/2, sz.height - 30, 60, 28);

		// Setup Quit window
		quitFrame.setResizable(false);
		quitFrame.setSize(370, 180);
		quitFrame.setLocation(200, 200);
		quitFrame.setLayout(null);
		Label ql1 = new Label("This program is launched and managed by Mathematica");
		Label ql2 = new Label("to support calling Java code from Mathematica.");
		Label ql3 = new Label("It is intended to be closed by calling UninstallJava[].");
		Label ql4 = new Label("You should not quit it manually unless you are sure you");
		Label ql5 = new Label("need to do so.");
		ql1.setFont(f);
		ql2.setFont(f);
		ql3.setFont(f);
		ql4.setFont(f);
		ql5.setFont(f);
		quitFrame.add(ql1);
		quitFrame.add(ql2);
		quitFrame.add(ql3);
		quitFrame.add(ql4);
		quitFrame.add(ql5);
		quitFrame.addNotify();
		in = quitFrame.getInsets();
		sz = quitFrame.getSize();
		ql1.setBounds(in.left + 10, in.top + 20, 360, 20);
		ql2.setBounds(in.left + 10, in.top + 40, 360, 20);
		ql3.setBounds(in.left + 10, in.top + 60, 360, 20);
		ql4.setBounds(in.left + 10, in.top + 80, 360, 20);
		ql5.setBounds(in.left + 10, in.top + 100, 360, 20);
		quitButton.addActionListener(listener);
		dontQuitButton.addActionListener(listener);
		quitFrame.add(dontQuitButton);
		quitFrame.add(quitButton);
		dontQuitButton.setBounds((sz.width - in.left - in.right - 200)/3, sz.height - 30, 100, 28);
		quitButton.setBounds(100 + 2*(sz.width - in.left - in.right - 200)/3, sz.height - 30, 100, 28);
	}
   
   
	public void handleAbout() {
		aboutFrame.setVisible(true);
		aboutFrame.toFront();
	}
	
	public void handleQuit() {
		quitFrame.setVisible(true);
		quitFrame.toFront();
	}
        
        
	class MRJActionListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
			Object source = e.getSource();
			if (source == okButton)
				aboutFrame.dispose();
			else if (source == dontQuitButton)
				quitFrame.dispose();
			else if (source == quitButton) {
				JLinkSecurityManager.setAllowExit(true);
				System.exit(0);
			}
		}
	}

    static class JLinkApplicationAdapter extends com.apple.eawt.ApplicationAdapter {

        MRJHandlers hndlrs;
        JLinkApplicationAdapter(MRJHandlers hndlrs) { this.hndlrs = hndlrs; }
        public void handleAbout(com.apple.eawt.ApplicationEvent evt) {
            hndlrs.handleAbout();
            evt.setHandled(true);
        }
        public void handleQuit(com.apple.eawt.ApplicationEvent evt) {
            hndlrs.handleQuit();
        }
    }

} 

