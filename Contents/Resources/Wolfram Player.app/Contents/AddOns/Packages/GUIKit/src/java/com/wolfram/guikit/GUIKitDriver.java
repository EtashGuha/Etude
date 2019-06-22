/*
 * @(#)GUIKitDriver.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 *
 * Based on models from BML, Bean Markup Language and other XML-GUI bridges
 */
package com.wolfram.guikit;

import java.io.*;
import java.lang.reflect.Method;
import java.net.URL;
import java.util.Hashtable;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TimerTask;
import java.util.Vector;
import java.awt.*;
import java.awt.event.*;

// Do we make a GUIKitAWTDriver with no swing code for people
// to use in things like Applets or non-swing environments?
// Maybe, but the default could/should support Swing

import javax.swing.*;
import javax.swing.border.Border;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.*;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

import com.wolfram.guikit.layout.GUIKitLayout;
import com.wolfram.guikit.layout.GUIKitLayoutInfo;
import com.wolfram.guikit.type.GUIKitTypedObject;
import com.wolfram.guikit.type.GUIKitTypedObjectProducer;
import com.wolfram.guikit.util.*;
import com.wolfram.guikit.swing.GUIKitJDialog;
import com.wolfram.guikit.swing.GUIKitJFrame;
import com.wolfram.bsf.util.concurrent.InvokeMode;
import com.wolfram.bsf.util.type.TypedObjectFactory;
import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.MathematicaBSFManager;
import com.wolfram.bsf.util.MathematicaEngineUtils;
import com.wolfram.bsf.util.MathematicaMethodUtils;
import com.wolfram.bsf.util.MathematicaObjectRegistry;

import com.wolfram.jlink.Expr;
import com.wolfram.jlink.KernelLink;
import com.wolfram.jlink.MathLinkException;
import com.wolfram.jlink.StdLink;


/**
 * GUIKitDriver
 */
public class GUIKitDriver implements WindowListener {

  public static final int RELEASE_ONCLOSE = 1;
  public static final int RELEASE_MANUAL = 2;

  public static final String ID_ROOTOBJECT = MathematicaBSFEngine.ID_PRIVATE_PREFIX + "Mathematica_RootObject";
  public static final String ID_GUIKITENV = MathematicaBSFEngine.ID_PRIVATE_PREFIX + "Mathematica_GUIKitEnvironment";
  
  public static final String ROOTOBJECT_SYMBOL = "Private`BSF`rootObject";
  
  protected static final int NO_RESULTMASK = 0;
  protected static final int DEFAULT_RESULTMASK = 1;
  protected static final int ARGGROUP_RESULTMASK = 2;
  
  // How complicated should urlCaching be, for now see about
  // a shared static cache and see where some might expect this
  // more dynamic per driver
  //private Hashtable urlCache = new Hashtable();
  private static Hashtable urlCache = new Hashtable();

  private static DocumentBuilderFactory documentFactory;
  
  // This will be the base environment for each driver instance
  // That may have temporary children environments during evaluations
  // TODO may consider having a persistent static parent for all drivers
  // that is created and saved for persistent storage
  protected GUIKitEnvironment env;

  /* These attributes can be set once or multiple times
     and their values are used with each run() creating a driver
     and environment based on these settings
   */
  protected KernelLink defaultLink = null;
  protected String defaultLinkCommandLine = null;
  protected String endModalScriptString = null;
  protected String[] additionalScriptContexts = null;
  protected String defaultContext = null;
  protected boolean debug = false;
  protected boolean useJavaShow = false;

  protected GUIKitTypedObject[] scriptArgs = null;

  protected boolean calledRequestEndModal = false;

  private boolean isRunning = false;
  private int executeReleaseMode = RELEASE_ONCLOSE;
  private boolean executeAsModal = false;
  
  private boolean externalEndModal = false;
  
  private Stack documentURLStack = new Stack();
  
  protected static Image frameImage = null;
  protected static boolean shouldSetInitialLookAndFeel = true;
  
  protected static java.util.Timer modalInterruptTimer = null;
  protected static ModalInterruptTimerTask modalInterruptTimerTask = null;
  
  static {
    TypedObjectFactory.setProducer(new GUIKitTypedObjectProducer());
    }
    
  public GUIKitDriver() {
    }

  public int getReleaseMode() {return executeReleaseMode;}
  protected void setReleaseMode(int m) {
    executeReleaseMode = m;
    }
    
  public boolean getExecuteAsModal() {return executeAsModal;}
  protected void setExecuteAsModal(boolean m) {
    executeAsModal = m;
    }
    
  public boolean getIsRunning() {return isRunning;}
  protected void setIsRunning(boolean newVal) {
    isRunning = newVal;
    }

  public boolean getUseJavaShow() {return useJavaShow;}
  public void setUseJavaShow(boolean useJavaShow) {
    this.useJavaShow = useJavaShow;
    }

  public KernelLink getLink() {return defaultLink;}
  public void setLink(KernelLink link) {
    this.defaultLink = link;
    }

  public String getLinkCommandLine() {return defaultLinkCommandLine;}
  public void setLinkCommandLine(String defaultLinkCommandLine) {
    this.defaultLinkCommandLine = defaultLinkCommandLine;
    }

  public String getReturnScript() {return endModalScriptString;}
  public void setReturnScript(String endModalScriptString) {
    this.endModalScriptString = endModalScriptString;
    registerObject(MathematicaBSFEngine.ID_ENDMODALSCRIPT, endModalScriptString,
       MathematicaObjectRegistry.SCOPE_OBJECT);
    externalEndModal = true;
    }

  public String[] getAdditionalScriptContexts() {return additionalScriptContexts;}
  public void setAdditionalScriptContexts(String[] contexts) {
    this.additionalScriptContexts = contexts;
    registerObject(MathematicaBSFEngine.ID_INCLUDEDSCRIPTCONTEXTS, contexts,
       MathematicaObjectRegistry.SCOPE_OBJECT);
    }
    
  public boolean getDebug() {return debug;}
  public void setDebug(boolean debug) {
    this.debug = debug;
    if (getGUIKitEnvironment() != null) {
      getGUIKitEnvironment().setDebug(debug);
      }
    }

  public String getContext() {return defaultContext;}
  public void setContext(String defaultContext) {
    this.defaultContext = defaultContext;
    }

  protected Object resolveExecuteObject(Object sourceObject) {
    Object rootObject = null;

    if ((sourceObject instanceof Component)) {
      if (sourceObject instanceof Window) {
        rootObject = sourceObject;
        }
      else if (sourceObject instanceof JComponent) {
        rootObject = new GUIKitJFrame();
        ((JFrame)rootObject).getContentPane().add((JComponent)sourceObject, BorderLayout.CENTER);
        }
      else {
        rootObject = new Frame();
        ((Frame)rootObject).add((Component)sourceObject, BorderLayout.CENTER);
        }
      }
    else rootObject = sourceObject;

    return rootObject;
    }

  protected Object resolveExecuteModalObject(Object sourceObject) {
    Object rootObject = null;

    if ((sourceObject instanceof Component)) {
      if (sourceObject instanceof Window) {
        rootObject = sourceObject;
        }
      else if (sourceObject instanceof JComponent) {
        rootObject = new GUIKitJFrame();
        ((JFrame)rootObject).getContentPane().add((JComponent)sourceObject, BorderLayout.CENTER);
        }
      else {
        rootObject = new Frame();
        ((Frame)rootObject).add((Component)sourceObject, BorderLayout.CENTER);
        }
      }
    // NOTE for modal call we will not by default set source to root unless
    // it correctly sets up an object that will call EndModal properly

    return rootObject;
    }

  // Consider registering this object with environment and
  // have default Frame check and set icon if Null
  // This would allow secondary frames to get default icon.
  // Also consider making default Dialog instances use a Frame
  // with this icon set so they will use Mathematica icons
  
  public static Image getDefaultFrameImage() {
    return getDefaultFrameImage(null);
    }
  
  public static Image getDefaultFrameImage(GUIKitDriver instance) {
    if (frameImage == null && instance != null) {
      String versionIcon = "5.0";
      MathematicaBSFEngine engine = instance.getMathematicaEngine();
      if (engine != null) {
        Double versionNumber = engine.getMathematicaVersionNumber();
        if (versionNumber != null && versionNumber.doubleValue() <= 4.2)
          versionIcon = "4.2";
        else if (versionNumber != null && versionNumber.doubleValue() >= 6.0)
          versionIcon = "6.0";
        }
      URL imageURL = GUIKitDriver.class.getClassLoader().getResource(
        "images/" + versionIcon + "/document32.gif");
      if (imageURL != null)
        frameImage = new ImageIcon(imageURL).getImage();
        
      GUIKitJDialog.setSharedFrameIconImage(frameImage);
      }
    if (frameImage != null && instance != null) {
      instance.registerObject(MathematicaBSFEngine.ID_DEFAULTFRAMEIMAGE, frameImage,
        MathematicaObjectRegistry.SCOPE_OBJECT);
      }
    return frameImage;
    }
  
  protected void prepareGUIKit(final Window win) {
    if (win instanceof Frame && ((Frame)win).getIconImage() == null) {
      if (frameImage != null)
        ((Frame)win).setIconImage(frameImage);
      }
    // here is where we can check if an existing listener exists for Meta-W for
    // window close request, if not add one that will initiate the event
    if (win instanceof JFrame) {
      // Check if a binding already exists on the root pane and if not add one
      KeyStroke closeStroke = KeyStroke.getKeyStroke(KeyEvent.VK_W, Toolkit.getDefaultToolkit().getMenuShortcutKeyMask());
      JRootPane rootPane = ((JFrame)win).getRootPane();
      if (KeyUtils.find(closeStroke, rootPane) == null) {
        rootPane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).put(
          closeStroke, "window.close");
        rootPane.getActionMap().put("window.close", new CloseAction(win));   
        }
      }
    // it should be quite rare to have an AWT Frame as the root window all should be JFrame
    else {
      win.addKeyListener(new KeyAdapter() {
        public void keyPressed(KeyEvent e) {
          if (e.getKeyCode() == KeyEvent.VK_W && 
              e.getModifiers() == Toolkit.getDefaultToolkit().getMenuShortcutKeyMask()) {
            requestClose(win);
            }
          }
        });
      
      }
    }

  public Object execute(Object executeObject, int releaseMode, boolean attemptResolve) {
    return execute(attemptResolve ? resolveExecuteObject(executeObject) : executeObject, releaseMode);
    }

  public Object execute(final Object executeObject, int releaseMode) {

    if (executeObject != null) {

      if (executeObject instanceof Window) {

        // TODO We might want to issue a special warning here or an error
        // will be printed because of failure?
				if (!requestSharedKernelState()) {
          destroyEnvironment();
          return null;
          }
				
        if (getGUIKitEnvironment() != null) {
          // These could be combined as one call without registering in the objectRegistry
          // but this could perhaps be useful elsewhere
          registerObject(ID_ROOTOBJECT, executeObject, MathematicaObjectRegistry.SCOPE_OBJECT);
          requestDeclareSymbol(ROOTOBJECT_SYMBOL, executeObject);
          }
          
        prepareGUIKit((Window)executeObject);
        boolean needsPack = false;
        
        if (!getIsRunning()) {
          setIsRunning(true);
          setReleaseMode(releaseMode);
          setExecuteAsModal(false);
          ((Window)executeObject).addWindowListener(this);
          
          if (((Window)executeObject).getLayout() != null)
            needsPack = true;
          }
          
        // Would be nice to call JavaShow if it is available
        Runnable r = new WindowShower((Window)executeObject, useJavaShow, needsPack);
        SwingUtilities.invokeLater(r);

        }
      else {
      	// rootObject was not a valid root for doing non-modal
        destroyEnvironment();
        }

      }
    else {
      destroyEnvironment();
      }
    return executeObject;
    }

  public boolean executeModal(Object executeObject, int releaseMode, boolean attemptResolve, boolean checkModalInterrupt) {
    return executeModal(attemptResolve ? resolveExecuteModalObject(executeObject) : executeObject, releaseMode, checkModalInterrupt);
    }

  public boolean executeModal(final Object executeObject, int releaseMode, boolean checkModalInterrupt) {
    boolean attemptModal = false;
    
    if (executeObject != null) {

      if (executeObject instanceof Window) {
        Window win = (Window)executeObject;
        attemptModal = true;

        if (getGUIKitEnvironment() != null) {
          // These could be combined as one call without registering in the objectRegistry
          // but this could perhaps be useful elsewhere
          registerObject(ID_ROOTOBJECT, executeObject, MathematicaObjectRegistry.SCOPE_OBJECT);
          requestDeclareSymbol(ROOTOBJECT_SYMBOL, executeObject);
          }
          
        prepareGUIKit((Window)executeObject);
          
        boolean needsPack = false;
        
        if (!getIsRunning()) {
          
          calledRequestEndModal = false;
          setIsRunning(true);
          setReleaseMode(releaseMode);
          setExecuteAsModal(true);
          win.addWindowListener(this);
          
          if (win.getLayout() != null)
            needsPack = true;
          
          // Setup the modal interrupt timer
          if (checkModalInterrupt) {
            modalInterruptTimerTask = new ModalInterruptTimerTask(this, executeObject);
            modalInterruptTimer = new java.util.Timer();
            modalInterruptTimer.schedule(modalInterruptTimerTask, 500, 1000);
            }
          }
        else {
          // TODO if already running (possibly non-modal, we can show it but
          // fail modal request, need to be sure existing run is still ok
          attemptModal = false;
          }
          
        // Nice to call JavaShow if it is available
        Runnable r = new WindowShower((Window)executeObject, useJavaShow, needsPack);
        SwingUtilities.invokeLater(r);
        
        }
      else {
				// rootObject was not a valid root for doing modal
        destroyEnvironment();
        }

      }
    else {
      destroyEnvironment();
      }

    return attemptModal;
    }

  // Public method for requesting an external termination
  // of an execute Object. This code should do whatever
  // is required to trigger a shutdown of the executeObject
  // identical to what is setup at the time of execute
  // Should consider if code needs to run in another thread
  public void requestRelease(final Object executeObject) {
    if (getIsRunning()) {
      terminateExecute(executeObject, true);
      }
    else destroyEnvironment();
    }
  
  public void requestClose(final Object executeObject) {
    if (getIsRunning())
      terminateExecute(executeObject, false);
    }
    
  private void cleanupWindowConnections(Window win) {
    if (win == null) return;
    
    win.removeWindowListener(this);
    
    if (win instanceof JFrame) {
      // Check if our "window.close" exists and remove and cleanup if it does
      KeyStroke closeStroke = KeyStroke.getKeyStroke(KeyEvent.VK_W, Toolkit.getDefaultToolkit().getMenuShortcutKeyMask());
      JRootPane rootPane = ((JFrame)win).getRootPane();
      if (KeyUtils.find(closeStroke, rootPane) != null) {
        Object act = rootPane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).get(closeStroke);
        if (act != null && act instanceof String && ((String)act).equals("window.close")) {
          rootPane.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW).remove(closeStroke);
          Object actionObject = rootPane.getActionMap().get("window.close");   
          if (actionObject != null && actionObject instanceof CloseAction) {
            rootPane.getActionMap().remove("window.close");
            ((CloseAction)actionObject).setWindow(null);
            }
          }
        }
      }
    }
  
  protected void terminateExecute(final Object executeObject, final boolean complete) {
    if (executeObject == null) return;

    if (executeObject instanceof Window) {
      
      if (complete && getReleaseMode() == RELEASE_MANUAL) {
        cleanupWindowConnections((Window)executeObject);
        
        if (SwingUtilities.isEventDispatchThread()) {
          if (getExecuteAsModal()) {
            if (!calledRequestEndModal)
              requestEndModal(true);
            }
          ((Window)executeObject).dispose();
          destroyEnvironment();
          }
        else {
          SwingUtilities.invokeLater( new Runnable() {
            public void run() {
              if (getExecuteAsModal()) {
                if (!calledRequestEndModal)
                  requestEndModal(true);
                }
              ((Window)executeObject).dispose();
              destroyEnvironment();
              }
            }
            );
          }
        }
        
      else {
        try {
          if (SwingUtilities.isEventDispatchThread()) {
            ((Window)executeObject).dispatchEvent(
              new WindowEvent((Window)executeObject, WindowEvent.WINDOW_CLOSING));
            }
          else {
            SwingUtilities.invokeLater( new Runnable() {
              public void run() {
                ((Window)executeObject).dispatchEvent(
                  new WindowEvent((Window)executeObject, WindowEvent.WINDOW_CLOSING));
                }
              }
              );
          
            }
          }
        catch (Exception e) {}
        }
        
      }

  }

  public void windowClosing(final WindowEvent e) {
    // Changes from commented code to not call
    // requestEndModal in the closing event
    // but only on windowClosed or componentHidden
    
    if (getReleaseMode() == RELEASE_MANUAL) {
      ((Window)e.getSource()).hide();
      SwingUtilities.invokeLater( new Runnable() {
        public void run() {
          initiateShutdown((Window)e.getSource());
          }
        });
      }
    else {
      ((Window)e.getSource()).dispose();
      }
    
    /*
    if (getExecuteAsModal()) {
      if (getReleaseMode() == RELEASE_MANUAL) {
        requestEndModal(false);
        ((Window)e.getSource()).hide();
        cleanupWindowConnections((Window)e.getSource());
        setIsRunning(false);
        }
      else {
        requestEndModal(true);
        ((Window)e.getSource()).dispose();
        }
      }
    else {
      if (getReleaseMode() == RELEASE_MANUAL) {
        ((Window)e.getSource()).hide();
        cleanupWindowConnections((Window)e.getSource());
        requestUnsharedKernelState();
        setIsRunning(false);
        }
      else {
        ((Window)e.getSource()).dispose();
        } 
      }
     */
    }
    
  protected void initiateShutdown(Window executeWindow) {
    if (executeWindow == null) return;
    if (getExecuteAsModal()) {
      if (getReleaseMode() == RELEASE_MANUAL) {
        if (!calledRequestEndModal) {
          requestEndModal(false);
          }
        if(getIsRunning()) {
          cleanupWindowConnections(executeWindow);
          setIsRunning(false);
          }
        }
      else {
         if (!calledRequestEndModal)
            requestEndModal(true);
          cleanupWindowConnections(executeWindow);
          destroyEnvironment();
        }
      }
    else {
      if (getReleaseMode() == RELEASE_MANUAL) {
        if(getIsRunning()) {
          cleanupWindowConnections(executeWindow);
          requestUnsharedKernelState();
          setIsRunning(false);
          }
        }
      else {
        cleanupWindowConnections(executeWindow);
        destroyEnvironment();
        }
      }
    }

  public void windowClosed(WindowEvent e) {
    initiateShutdown((Window)e.getSource());
    }        
                
  public void windowOpened(WindowEvent e) {}
  public void windowIconified(WindowEvent e) {}
  public void windowDeiconified(WindowEvent e) {}
  public void windowActivated(WindowEvent e) {}
  public void windowDeactivated(WindowEvent e) {}
  
  public void destroyEnvironment() {
    destroyEnvironment(null);
    }

  public void destroyEnvironment(Exception e) {

    if (getIsRunning()) {
      setIsRunning(false);
      }

    // This will call terminate on all engines in the manager
    if (getGUIKitEnvironment() != null) {
    	getGUIKitEnvironment().destroy();
      getGUIKitEnvironment().setDriver(null);
      }

    // This may need to come back on but see if java ref is taken 
    //  care of in undeclares
		//requestReleaseJavaObject(this);
		
    // After making sure objects gc make sure we do not need to explicitly null
    // out environment ivars too

    // Might want to clear out objectRegistry for instance and null out
    // GUIKitEnvironment ivars

    setGUIKitEnvironment(null);
    }

  // Make multiple variants supporting min/max scope
  public Object lookupObject(String id) {
    return lookupObject(id, MathematicaObjectRegistry.SCOPE_FIRST, MathematicaObjectRegistry.SCOPE_LAST);
    }
  public Object lookupObject(String id, int maxScope) {
   return lookupObject(id, MathematicaObjectRegistry.SCOPE_FIRST, maxScope);
   }
   
  public Object lookupObject(String id, int minScope, int maxScope) {
    if (id == null || getGUIKitEnvironment() == null) return null;
    return getGUIKitEnvironment().lookupObject(id, minScope, maxScope);
    }
  
  public void setScriptArguments(Vector args) {
    GUIKitTypedObject[] useArgs = null;
    if (args != null) {
      useArgs = new GUIKitTypedObject[args.size()];
      for (int i = 0; i < args.size(); ++i) {
        Object obj = args.elementAt(i);
        if (obj != null) {
          if (obj instanceof GUIKitTypedObject) useArgs[i] = (GUIKitTypedObject)obj;
          else useArgs[i] = (GUIKitTypedObject)TypedObjectFactory.create(obj);
          }
        else useArgs[i] = null;
        }
      }
    setScriptArguments(useArgs);
    }

 public void setScriptArguments(GUIKitTypedObject[] newargs) {
    scriptArgs = newargs;
    if (getGUIKitEnvironment() != null) {
      // TODO This set of arguments should stay around through the lifetime of the definition
      // and be visible to all components especially the top level one
      // Since real ACTION calls need to override this visibility this
      // Should probably be created in a SCOPE_OBJECT
      if (scriptArgs != null)
        GUIKitUtils.registerAsScopeArguments(getGUIKitEnvironment(), scriptArgs, MathematicaObjectRegistry.SCOPE_OBJECT);
      }
    }
  public GUIKitTypedObject[] getScriptArguments() {
    return scriptArgs;
    }

  public void registerObject(String id, Object obj) {
    registerObject(id, obj, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }
  public void registerObject(String id, Object obj, int scope) {
    if (id == null || getGUIKitEnvironment() == null) return;
    if (obj == null) unregisterObject(id, scope);
    // When talking to driver user registers should go to manager's object registry by default
    else getGUIKitEnvironment().registerObject(id, obj, scope);
    }

  public void unregisterObject(String id) {
    unregisterObject(id, MathematicaObjectRegistry.SCOPE_DEFAULT);
    }
  public void unregisterObject(String id, int scope) {
    if (id == null || getGUIKitEnvironment() == null) return;
    // When talking to driver user registers should go to manager's object registry by default
    getGUIKitEnvironment().unregisterObject(id, scope);
    }

  public static void prepareInitialLookAndFeel() {
    /* Current plan is:
     * 
     * For Java VMs 1.4.1 or less:
     *   Mac OS X : use default Aqua look and feel
     *   Linux: use default Java look and feel (looks better than Motif)
     *   Windows: change to System look and feel (better than default Java)
     * 
     * For Java VMs 1.4.2 or greater
     *   Mac OS X : use default Aqua look and feel
     *   Linux: GTK LAF
     *   Windows: change to System look and feel which also will pick up XP LAF 
     */
    try {
      if (com.wolfram.jlink.Utils.isWindows()) {
        try {
          String desiredLAF = UIManager.getSystemLookAndFeelClassName();
          String currentLAF = null;
          if (UIManager.getLookAndFeel() != null) {
            currentLAF = UIManager.getLookAndFeel().getClass().getName();
            }
          if (!desiredLAF.equals(currentLAF))
            UIManager.setLookAndFeel(desiredLAF);
          }
        catch (Exception ei) {}

        // This is currently done because Windows LAF in 1.4.x does not show any
        // indication of focus on the default cell renderer for Object.class
        // TODO need to see if Windows XP is ok with the dashed default border and only 2000 is not
        Color selColor = UIManager.getDefaults().getColor("Table.selectionBackground");
        UIManager.getDefaults().put("Table.focusCellHighlightBorder", 
          BorderFactory.createMatteBorder(1,1,1,1, selColor));
        }
      else if (!com.wolfram.jlink.Utils.isMacOSX()) {
          // tgayley: The following should work to set the "best" look and feel on Linux, but it makes Java crash (bug 279331):
          //    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
          // Similarly, the old code in this block, which manually set "com.sun.java.swing.plaf.gtk.GTKLookAndFeel", caused
          // the same crash. Doing nothing here works fine.
      }
    } catch (Exception e) {}
    }
    
  protected void createEnvironment() {
  	// Preexisting environments will be used when createEnvironment is called
  	// as this helps when calling GUIKitResolve with existing drivers
  	if (getGUIKitEnvironment() != null) return;
  		
    // Currently we decide to set the
    // look and feel for platforms *once* which allows
    // a user to change this and we will not force this
    // choice each time an instance is loaded
    
    if (shouldSetInitialLookAndFeel) {
      prepareInitialLookAndFeel();
      shouldSetInitialLookAndFeel = false;
      }

    GUIKitEnvironment env = new GUIKitEnvironment(null, MathematicaObjectRegistry.SCOPE_OBJECT);
    setGUIKitEnvironment(env);
    env.setDebug(debug);

    initEnvironment();
    }

  protected void initEnvironment() {
    registerObject(MathematicaBSFEngine.ID_KERNELLINK, getLink(), MathematicaObjectRegistry.SCOPE_OBJECT);
    registerObject(MathematicaBSFEngine.ID_KERNELLINKCOMMANDLINE, getLinkCommandLine(), MathematicaObjectRegistry.SCOPE_OBJECT);
    registerObject(MathematicaBSFEngine.ID_CONTEXT, getContext(), MathematicaObjectRegistry.SCOPE_OBJECT);
    registerObject(MathematicaBSFEngine.ID_ENDMODALSCRIPT, getReturnScript(), MathematicaObjectRegistry.SCOPE_OBJECT);
    registerObject(MathematicaBSFEngine.ID_INCLUDEDSCRIPTCONTEXTS, getAdditionalScriptContexts(), MathematicaObjectRegistry.SCOPE_OBJECT);
    registerObject(MathematicaBSFEngine.ID_DRIVER, this, MathematicaObjectRegistry.SCOPE_OBJECT);
    
    // Also setup the default frame icon once here
    getDefaultFrameImage(this);
    }
  
  protected GUIKitTypedObject load(Object docSource, URL contextURL) throws GUIKitException {

    if (contextURL != null)
      getGUIKitEnvironment().registerObject(MathematicaBSFEngine.ID_SCRIPTCONTEXTURL, contextURL,
        MathematicaObjectRegistry.SCOPE_OBJECT);

    if (documentURLStack != null)
      getGUIKitEnvironment().registerObject(MathematicaBSFEngine.ID_SCRIPTURLSTACK, documentURLStack,
        MathematicaObjectRegistry.SCOPE_OBJECT);
        
    Document doc = null;

    if (docSource instanceof Document) {
      doc = (Document)docSource;
      }
    else if (docSource instanceof URL) {
      try {
        doc = parse((URL)docSource);
        }
      catch (Exception ie) {
        throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, 
				(ie.getMessage() != null) ? ie.getMessage() : ie.getClass().getName());
        }
      }
    else if (docSource instanceof Reader) {
      try {
        doc = parse((Reader)docSource);
        }
      catch (Exception ie) {
        throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, 
				(ie.getMessage() != null) ? ie.getMessage() : ie.getClass().getName());
        }
      }

    // Might want to throw an exception
    if (doc == null) return null;

    // In case scriptArgs were set before GUIKitEnvironment existed we make
    // sure they are set now
    setScriptArguments(scriptArgs);
    
    return processDocument(getGUIKitEnvironment(), doc, null, contextURL, null);
    }

  public boolean requestSharedKernelState() {
    boolean calledShareKernel = true;
    MathematicaBSFEngine engine = getMathematicaEngine();
    if (engine != null) {
      try {
        calledShareKernel = engine.shareKernel();
        }
      catch (MathLinkException me) {me.printStackTrace();}
      }
    return calledShareKernel;
    }

  public void requestUnsharedKernelState() {
    MathematicaBSFEngine engine = getMathematicaEngine();
    if (engine != null) {
      engine.unshareKernel();
      }
    }
    
  // This may not need to be public
  public void requestEndModal(boolean cleanup) {
    MathematicaBSFEngine engine = getMathematicaEngine();
    if (engine != null) {
      boolean wasInterrupted = false;
      // stop the modal interrupt timer
      if (modalInterruptTimer != null) {
        modalInterruptTimer.cancel();
        modalInterruptTimer = null;
        if (modalInterruptTimerTask != null) {
          wasInterrupted = modalInterruptTimerTask.getWasInterrupted();
          modalInterruptTimerTask.cancel();
          modalInterruptTimerTask = null;
          }
        }
      
      Object result = null;
      try {
        if (wasInterrupted) {
          result = new Expr(Expr.SYMBOL, "$Aborted");
          }
        else {
          result = engine.evaluateEndModalResult();
          }
      	if (cleanup) {
					if (getGUIKitEnvironment() != null) {
						getGUIKitEnvironment().destroyChildrenEnvironments();
						}
      		engine.cleanupContext();
      		}
   			engine.processEndModal(result);
        }
      catch (MathLinkException me) {me.printStackTrace();}
      }
    calledRequestEndModal = true;
    }

  // TODO consider whether public declare symbol optionally
  // adds an argument for context
  public void requestDeclareSymbol(String sym, Object obj) {
    MathematicaBSFEngine engine = getMathematicaEngine();
    if (engine != null) {
      engine.requestDeclareSymbol(sym, obj);
      }
    }

  public void requestReleaseJavaObject(Object obj) {
    MathematicaBSFEngine engine = getMathematicaEngine();
    if (engine != null) {
      try {
        engine.requestReleaseJavaObject(obj);
        }
      catch (MathLinkException me) {me.printStackTrace();}
      }
    }

  public void requestJavaShow(Object window) {
    MathematicaBSFEngine engine = getMathematicaEngine();
    if (engine != null) {
      try {
        engine.requestJavaShow(window);
        }
      catch (MathLinkException me) {me.printStackTrace();}
      }
    }

  private MathematicaBSFEngine getMathematicaEngine() {
    if (getGUIKitEnvironment() != null)
      return getGUIKitEnvironment().getMathematicaEngine();
    return null;
    }
    
  public void requestHandleException(Exception e, MathematicaBSFEngine engine) {

    if (getGUIKitEnvironment() != null && getGUIKitEnvironment().getDebug()) {
      e.printStackTrace( getGUIKitEnvironment().getDebugStream());
      }

    if (engine != null) {
      try {
        engine.requestHandleException(e);
        }
      catch (MathLinkException me) {me.printStackTrace();}
      }
    }

  private Document resolveGUIKitURL(URL contextURL) throws GUIKitException {
    Document result = null;
    try {
      String xmlString = null;
      // We check for file protocol and request an eval in Mathematica that does a Get on file
      if (contextURL != null && "file".equals(contextURL.getProtocol())) {
        //System.out.println("File Get resolveGUIKitURL: " + contextURL.getPath());
        xmlString = getMathematicaEngine().requestExprFileToXMLString(
           contextURL.getPath(), GUIKitUtils.GUI_XMLFORMAT);
        }
      else {
        String guikitString = MathematicaEngineUtils.getContentAsString(contextURL);
        xmlString = getMathematicaEngine().requestExprToXMLString(
           guikitString, GUIKitUtils.GUI_XMLFORMAT);
        }
      result = parse(new StringReader(xmlString));
      }
    catch (Exception ie) {
      throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR,
			(ie.getMessage() != null) ? ie.getMessage() : ie.getClass().getName());
      }
    return result;
    }

  public void handleException(Exception e) {
		handleException(e, getMathematicaEngine());
  }
  
  protected void handleException(Exception e, MathematicaBSFEngine engine) {
    // By default we will assume the driver/engine is running within Mathematica
    // and only return exceptions as messages
    // we need to make a request to the engine to print the appropriate
    // message and probably manually return $Failed
    requestHandleException(e, engine);
    }

  public Object runFile(String file) {
    return runFile(file, RELEASE_ONCLOSE);
    }
    
  public Object runFile(String file, int releaseMode) {
    Object resultObj = null;
    boolean needsDestroyEnvironment = false;

    try {
      createEnvironment();

      URL contextURL = MathematicaEngineUtils.getMathematicaURL(
        GUIKitEnvironment.RESOLVE_FUNCTION, null, file, getMathematicaEngine(),
        getURLCache());
      GUIKitTypedObject sourceObject = null;

      // pass in resolved Document to first argument here if .m
      try {
        if (contextURL != null && contextURL.getPath() != null &&
            contextURL.getPath().endsWith(".m")) {
          sourceObject = load( resolveGUIKitURL(contextURL), contextURL);
          }
        else {
          sourceObject = load( contextURL, contextURL);
          }
        }
      catch (GUIKitException ie) {
        needsDestroyEnvironment = true;
        throw ie;
        }

      resultObj = execute( sourceObject.value, releaseMode, true);
      }
    catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (needsDestroyEnvironment)
        destroyEnvironment(e);
      handleException(e, engine);
      }

    return resultObj;
    }

  public Object runContent(String xmlContent) {
    return runContent(xmlContent, RELEASE_ONCLOSE);
    }

  public Object runContent(String xmlContent, int releaseMode) {
    return runContent(xmlContent, releaseMode, null);
    }
    
  public Object runContent(String xmlContent, int releaseMode, String contextPath) {
    Object resultObj = null;
    boolean needsDestroyEnvironment = false;

    try {
      URL contextURL = null;

      createEnvironment();

      // Need to decide whether here or in BSF engine do we allow $Path to include the
      // directory of this contextPath/file/dir

      if (contextPath != null) {
        contextURL = MathematicaEngineUtils.getMathematicaURL(
          GUIKitEnvironment.RESOLVE_FUNCTION, null, contextPath, getMathematicaEngine(),
          getURLCache());
        }

      GUIKitTypedObject sourceObject = null;

      try {
        sourceObject = load( new StringReader(xmlContent), contextURL);
        }
      catch (GUIKitException ie) {
        needsDestroyEnvironment = true;
        throw ie;
        }

      resultObj = execute( sourceObject.value, releaseMode, true);
     }
    catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (needsDestroyEnvironment)
        destroyEnvironment(e);
      handleException(e, engine);
      }

    return resultObj;
    }

  public boolean runModalFile(String file, boolean checkModalInterrupt) {
    return runModalFile(file, RELEASE_ONCLOSE, checkModalInterrupt);
    }
    
  public boolean runModalFile(String file, int releaseMode, boolean checkModalInterrupt) {
    boolean result = false;
    boolean needsDestroyEnvironment = false;

    try {
      createEnvironment();

      URL contextURL = MathematicaEngineUtils.getMathematicaURL(GUIKitEnvironment.RESOLVE_FUNCTION,
        null, file, getMathematicaEngine(), getURLCache());
      GUIKitTypedObject sourceObject = null;

      // pass in resolved Document object here in first argument
      try {
        if (contextURL != null && contextURL.getPath() != null &&
            contextURL.getPath().endsWith(".m")) {
          sourceObject = load( resolveGUIKitURL(contextURL), contextURL);
          }
        else {
          sourceObject = load( contextURL, contextURL);
          }
        }
      catch (GUIKitException ie) {
        needsDestroyEnvironment = true;
        throw ie;
        }

      result = executeModal( sourceObject.value, releaseMode, true, checkModalInterrupt);
      }
     catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (needsDestroyEnvironment)
        destroyEnvironment(e);
      handleException(e, engine);
      }

    return result;
    }

  public boolean runModalContent(String xmlContent, boolean checkModalInterrupt) {
    return runModalContent(xmlContent, RELEASE_ONCLOSE, checkModalInterrupt);
    }

  public boolean runModalContent(String xmlContent, int releaseMode, boolean checkModalInterrupt) {
    return runModalContent(xmlContent, releaseMode, null, checkModalInterrupt);
    }
    
  public boolean runModalContent(String xmlContent, int releaseMode, String contextPath, boolean checkModalInterrupt) {
    boolean result = false;
    boolean needsDestroyEnvironment = false;

    try {
      URL contextURL = null;

      createEnvironment();
      // Need to decide whether here or in BSF engine do we allow $Path to include the
      // directory of this contextPath/file/dir

      if (contextPath != null) {
        contextURL = MathematicaEngineUtils.getMathematicaURL(GUIKitEnvironment.RESOLVE_FUNCTION,
          null, contextPath, getMathematicaEngine(), getURLCache());
        }

      GUIKitTypedObject sourceObject = null;

      try {
        sourceObject = load( new StringReader(xmlContent), contextURL);
        }
      catch (GUIKitException ie) {
        needsDestroyEnvironment = true;
        throw ie;
        }

      result = executeModal( sourceObject.value, releaseMode, true, checkModalInterrupt);
      }
     catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (needsDestroyEnvironment)
        destroyEnvironment(e);
      handleException(e, engine);
      }

    return result;
    }

  public Object loadFile(String file) {
    return loadFile(file, true);
    }
    
  public Object loadFile(String file, boolean shouldCreateEnvironment) {
    GUIKitTypedObject result = loadFileObject(file, shouldCreateEnvironment);
    if (result != null) return result.value;
    else return null;
    }
  	
  protected GUIKitTypedObject loadFileObject(String file, boolean shouldCreateEnvironment) {
    GUIKitTypedObject sourceObject = null;
    boolean needsDestroyEnvironment = false;

    try {
    	if (shouldCreateEnvironment)
      	createEnvironment();

      URL contextURL = MathematicaEngineUtils.getMathematicaURL(GUIKitEnvironment.RESOLVE_FUNCTION,
        null, file, getMathematicaEngine(), getURLCache());

      // pass in resolved Document object to first argument
      try {
        if (contextURL != null && contextURL.getPath() != null &&
            contextURL.getPath().endsWith(".m")) {
          sourceObject = load( resolveGUIKitURL(contextURL), contextURL);
          }
        else {
          sourceObject = load( contextURL, contextURL);
          }
        }
      catch (GUIKitException ie) {
      	if (shouldCreateEnvironment)
        	needsDestroyEnvironment = true;
        throw ie;
        }
      }
    catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (needsDestroyEnvironment)
        destroyEnvironment(e);
      handleException(e, engine);
      }

    return sourceObject;
    }

  public Object loadContent(String xmlContent) {
    return loadContent(xmlContent, null);
    }

  public Object loadContent(String xmlContent, String contextPath) {
    return loadContent(xmlContent, contextPath, true);
    }
    
  public Object loadContent(String xmlContent, String contextPath, boolean shouldCreateEnvironment) {
    GUIKitTypedObject result = loadContentObject(xmlContent, contextPath, shouldCreateEnvironment);
    if (result != null) return result.value;
    else return null;
    }
  	
  protected GUIKitTypedObject loadContentObject(String xmlContent, String contextPath, boolean shouldCreateEnvironment) {
    GUIKitTypedObject sourceObject = null;
    boolean needsDestroyEnvironment = false;

    try {
      URL contextURL = null;

			if (shouldCreateEnvironment)
      	createEnvironment();
      	
      // Need to decide whether here or in BSF engine do we allow $Path to include the
      // directory of this contextPath/file/dir
      if (contextPath != null) {
        contextURL = MathematicaEngineUtils.getMathematicaURL(GUIKitEnvironment.RESOLVE_FUNCTION,
          null, contextPath, getMathematicaEngine(), getURLCache());
        }

      try {
        sourceObject = load( new StringReader(xmlContent), contextURL);
        }
      catch (GUIKitException ie) {
      	if (shouldCreateEnvironment)
        	needsDestroyEnvironment = true;
        throw ie;
        }
      }
    catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (needsDestroyEnvironment)
        destroyEnvironment(e);
      handleException(e, engine);
      }

    return sourceObject;
    }

  public Object resolveFile(String file) {
    return resolveFile(file, true);
    }
    
  public Object resolveFile(String file, boolean shouldCreateEnvironment) {
    GUIKitTypedObject result = resolveFileObject(file, shouldCreateEnvironment);
    if (result != null) return result.value;
    else return null;
    }
  
  public GUIKitTypedObject resolveFileObject(String file) {
  	return resolveFileObject(file, true);
  	}
  	
  public GUIKitTypedObject resolveFileObject(String file, boolean shouldCreateEnvironment) {
    GUIKitTypedObject result = null;

    try {
      result = loadFileObject(file, shouldCreateEnvironment);
      if (shouldCreateEnvironment)
      	destroyEnvironment();
      }
    catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (shouldCreateEnvironment)
      	destroyEnvironment(e);
      handleException(e, engine);
      }

    return result;
    }

  public Object resolveContent(String xmlContent) {
    return resolveContent(xmlContent, null);
    }

  public Object resolveContent(String xmlContent, boolean shouldCreateEnvironment) {
    return resolveContent(xmlContent, null, shouldCreateEnvironment);
    }
    
  public Object resolveContent(String xmlContent, String contextPath) {
    return resolveContent(xmlContent, contextPath, true);
    }
    
  public Object resolveContent(String xmlContent, String contextPath, boolean shouldCreateEnvironment) {
    GUIKitTypedObject result = resolveContentObject(xmlContent, contextPath, shouldCreateEnvironment);
    if (result != null) return result.value;
    else return null;
    }
  
  public GUIKitTypedObject resolveContentObject(String xmlContent) {
    return resolveContentObject(xmlContent, null);
    }

  public GUIKitTypedObject resolveContentObject(String xmlContent, boolean shouldCreateEnvironment) {
    return resolveContentObject(xmlContent, null, shouldCreateEnvironment);
    }
    
  public GUIKitTypedObject resolveContentObject(String xmlContent, String contextPath) {
  	return resolveContentObject(xmlContent, contextPath, true);
  	}
  	
  public GUIKitTypedObject resolveContentObject(String xmlContent, String contextPath, boolean shouldCreateEnvironment) {
    GUIKitTypedObject result = null;

    try {
      result = loadContentObject( xmlContent, contextPath, shouldCreateEnvironment);
      if (shouldCreateEnvironment)
      	destroyEnvironment();
      }
    catch (Exception e) {
      MathematicaBSFEngine engine = getMathematicaEngine();
      if (shouldCreateEnvironment)
      	destroyEnvironment(e);
      handleException(e, engine);
      }

    return result;
    }

  public GUIKitEnvironment getGUIKitEnvironment() {return env;}
  public void setGUIKitEnvironment(GUIKitEnvironment environment) {
    env = environment;
    if (env != null) {
      env.setDriver(this);
      }
    }
  
  public Document parse(URL contextURL) throws IOException,
     ParserConfigurationException, SAXException {
    Document doc = null;

    ClassLoader orig = Thread.currentThread().getContextClassLoader();

    /* This temp swapping of the thread context classloader is a workaround
      for when pre 1.4 VMs are used and the XML parser dynamic discovery
      doesn't use the JLink ClassLoader, (for instance in the EventDispatching thread )

      We may need to check that we only do this within a Mathematica use of JLink
      and not in standalone apps or webMathematica
      
      This is also currently needed for JavaCells given their initial classloading
    */
    Thread.currentThread().setContextClassLoader(
      getMathematicaEngine().getClassLoader());
      
    if(documentFactory == null) {
      documentFactory = DocumentBuilderFactory.newInstance();
      }
     
    try {
      doc = documentFactory.newDocumentBuilder().parse(contextURL.toExternalForm());
      }
    finally {
      Thread.currentThread().setContextClassLoader(orig);
      }

    return doc;
    }

  public Document parse(Reader r) throws IOException,
      ParserConfigurationException, SAXException {
    Document doc = null;

    ClassLoader orig = Thread.currentThread().getContextClassLoader();

    /* This temp swapping of the thread context classloader is a workaround
      for when pre 1.4 VMs are used and the XML parser dynamic discovery
      doesn't use the JLink ClassLoader, (for instance in the EventDispatching thread )

      We may need to check that we only do this within a Mathematica use of JLink
      and not in standalone apps or webMathematica.
      
      This is also currently needed for JavaCells given their initial classloading
    */
    Thread.currentThread().setContextClassLoader(
      getMathematicaEngine().getClassLoader());
      
    if(documentFactory == null) {
      documentFactory = DocumentBuilderFactory.newInstance();
      }
      
    try {
      doc = documentFactory.newDocumentBuilder().parse(new InputSource(r));
      }
    finally {
      Thread.currentThread().setContextClassLoader(orig);
      }

    return doc;
    }

  public Hashtable getURLCache() {return urlCache;}
  public void clearURLCache() {
    urlCache.clear();
    }
  
  private GUIKitTypedObject resolveObjectName(GUIKitEnvironment environ, String src, GUIKitTypedObject bean) throws GUIKitException {
    if(src == null || src.equals(GUIKitUtils.ATTVAL_THIS)) {
      if(bean != null)
        return bean;
      else return null;
      }
    
    if(src.startsWith("class:")) {
      return (GUIKitTypedObject)TypedObjectFactory.create(java.lang.Class.class, environ.resolveClassName(src.substring(6)));
      }

    Object obj = null;
    try {
      obj = environ.lookupObject(src);
      }
    catch(IllegalArgumentException ie) {
      throw new GUIKitException(GUIKitException.REASON_UNKNOWN_OBJECT, ie.getMessage());
      }
      
    if (obj == null) return null;
    
    return (GUIKitTypedObject)TypedObjectFactory.create(obj);
    }

  protected GUIKitTypedObject processNode(GUIKitEnvironment environ, Node node, 
      GUIKitTypedObject bean, URL url, GUIKitLayoutInfo parentLayoutInfo) throws GUIKitException {
    return processNode(environ, node, bean, url, parentLayoutInfo, DEFAULT_RESULTMASK);
    }
    
  protected GUIKitTypedObject processNode(GUIKitEnvironment environ, Node node, 
      GUIKitTypedObject bean, URL url, GUIKitLayoutInfo parentLayoutInfo, int resultMask) throws GUIKitException {
    short nodeType = node.getNodeType();

    if(nodeType == Node.TEXT_NODE || nodeType == Node.COMMENT_NODE)
        return null;

    if(nodeType == Node.ELEMENT_NODE) {
      Element element = (Element)node;
      String tagName = element.getTagName();

      if(tagName.equals(GUIKitUtils.ELEM_WIDGET))
        return processComponent(environ, element, bean, url, parentLayoutInfo);
      if(tagName.equals(GUIKitUtils.ELEM_GROUP))
        return processGroup(environ, element, bean, url, parentLayoutInfo, resultMask);
      if(tagName.equals(GUIKitUtils.ELEM_SCRIPT))
        return processScript(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_PROPERTY))
        return processProperty(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_STRING))
        return processString(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_INTEGER))
        return processInteger(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_DOUBLE))
        return processDouble(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_INVOKEMETHOD))
        return processInvokeMethod(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_BINDEVENT))
        return processBindEvent(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_NULL))
        return processNull(environ, element, bean, url);
      if(tagName.equals(GUIKitUtils.ELEM_TRUE))
        return GUIKitTypedObject.TYPED_TRUE;
      if(tagName.equals(GUIKitUtils.ELEM_FALSE))
        return GUIKitTypedObject.TYPED_FALSE;
      // This is processed in component or group calls
      if(tagName.equals(GUIKitUtils.ELEM_LAYOUT))
        return null;
      if(tagName.equals(GUIKitUtils.ELEM_SPACE)) {
        processSpace(environ, element, bean, url);
			  return null;
        }
      if(tagName.equals(GUIKitUtils.ELEM_FILL)) {
        processFill(environ, element, bean, url);
			  return null;
        }
      if(tagName.equals(GUIKitUtils.ELEM_ALIGN)) {
        processAlign(environ, element, bean, url);
			  return null;
        } 
      // This element is handled in code for processComponent
      if(tagName.equals(GUIKitUtils.ELEM_EXPOSE))
        return null;
      // This element is handled in code for processComponent
      if(tagName.equals(GUIKitUtils.ELEM_ARGS)) {
        return null;
        }
      else {
        System.err.println("WARNING: Unknown element '" + element + "' ignored.");
        return null;
        }
      }
    if(nodeType == Node.PROCESSING_INSTRUCTION_NODE) {
      processPI(environ, (ProcessingInstruction)node);
      return null;
      }
    else {
      System.err.println("WARNING: Unknown node '" + node + "' ignored.");
      return null;
      }
    }

  // Currently no known GUIKit processing instructions are defined or supported
  protected void processPI(GUIKitEnvironment environ, ProcessingInstruction processinginstruction) throws GUIKitException {
    String target = processinginstruction.getTarget();
    String data = processinginstruction.getData();

    if(!target.equalsIgnoreCase(GUIKitUtils.PI_GUI_XMLFORMAT))
      return;
    if(data == null || (data = data.trim()).equals(""))
      throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "bad data in PI <?" + target + " " + data + "?>");

    StringTokenizer stringtokenizer = new StringTokenizer(data);
    if(stringtokenizer.countTokens() % 2 != 0)
      throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "pi needs even number of arguments: " + target + " " + data);

    while(stringtokenizer.hasMoreTokens()) {
      //String piCommand
      stringtokenizer.nextToken();
      //String className
      stringtokenizer.nextToken();

      }
    }

  // TODO think about instead of using flag, pass in a Vector from arguments and if not null
  // add a bean result.  This can/is related to whether we can support general lists in definition
  // that turn into ComponentGroup and have Group work at returning its children beans
  // and work as arguments to <property> for example but still work with component layouts
  
  protected GUIKitTypedObject[] processChildren(GUIKitEnvironment environ, Element element, 
        GUIKitTypedObject bean, URL url, GUIKitLayoutInfo parentLayoutInfo, int resultMask) throws GUIKitException {
    Vector vector = (resultMask == 0) ? null : new Vector();

    for(Node node = element.getFirstChild(); node != null; node = node.getNextSibling()) {
      GUIKitTypedObject bean1 = processNode(environ, node, bean, url, parentLayoutInfo, resultMask);
      // Filter only potential argument generating nodes for results
      if(resultMask != 0 && (node.getNodeType() == Node.ELEMENT_NODE)) {
        // If a child fails to return an instance it should still have a NULL placeholder
        if (bean1 == null) bean1 = GUIKitTypedObject.TYPED_NULL;
        vector.addElement(bean1);
        bean1 = null;
        }
      }
    return (vector != null ? (GUIKitTypedObject[])vector.toArray(new GUIKitTypedObject[]{}) : null);
    }

  protected GUIKitTypedObject processProperty(GUIKitEnvironment environ, Element element, GUIKitTypedObject bean, URL url) throws GUIKitException {
    String target = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_TARGET);
    GUIKitTypedObject targetBean = resolveObjectName(environ, target, bean);
    String name = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_NAME);
    String index = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_INDEX);
    String valueString = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_VALUE);
    GUIKitTypedObject valueBean = null;
    boolean getFlag = false;

    if(valueString != null) {
      valueBean = (GUIKitTypedObject)TypedObjectFactory.create(java.lang.String.class, valueString);
      }
    else {
      GUIKitTypedObject[] childResult = processChildren(environ, element, bean, url, null, 
        ARGGROUP_RESULTMASK | DEFAULT_RESULTMASK);
      if(childResult.length == 0)
        getFlag = true;
      else
        valueBean = childResult[0];
      }
      
    Integer integerIndex = null;
    if(index != null)
      try {
        // Mathematica and XML 1-based, Java 0-based
        integerIndex = new Integer(Integer.parseInt(index) - 1);
        }
      catch(NumberFormatException _ex) {
        throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "non-integer value for '" + 
          GUIKitUtils.ATT_INDEX + "' attribute of <property> element");
        }
        
    GUIKitTypedObject resultBean = null;
    
    InvokeMode mode = GUIKitUtils.determineInvokeMode(
        GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_INVOKETHREAD), 
        GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_INVOKEWAIT) );
    if (environ != null) mode.setManager( environ.getBSFManager());
      
    if(getFlag) {
      valueBean = GUIKitUtils.getBeanProperty(environ, targetBean, name, integerIndex, mode);
      resultBean = valueBean;
      }
    else {
      GUIKitUtils.setBeanProperty(environ, targetBean, name, integerIndex, valueBean, mode);
      resultBean = null;
      }
      
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify this should be default scope
    if(id != null)
      environ.registerObject(id, valueBean.value);
      
    return resultBean;
    }

  protected GUIKitTypedObject processInvokeMethod(GUIKitEnvironment environ, Element element, GUIKitTypedObject bean, URL url) throws GUIKitException {
    String target = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_TARGET);
    GUIKitTypedObject targetBean = resolveObjectName(environ, target, bean);
    
    if (targetBean == null)
      throw new GUIKitException(GUIKitException.REASON_CALL_METHOD_ERROR, "The target, " + target + ", to the call method is unknown.");
    
    String methodName = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_NAME);
    Object argObjects[] = null;
    Class argClasses[] = null;
    
    GUIKitTypedObject[] childResult = processChildren(environ, element, bean, url, null, ARGGROUP_RESULTMASK | DEFAULT_RESULTMASK);
    
    Object argId = null; 
    int argLength = 0;
    if (childResult != null)
      argLength += childResult.length;
       
    // Next we check for the existence of an args child which would
    // populate/or supplement the arguments and possibly have a ref attribute that specified a
    // list of arguments
    Node node = null;
    for(node = element.getFirstChild(); node != null; node = node.getNextSibling())
       if(node.getNodeType() == Node.ELEMENT_NODE) break;
       
    if(node != null && ((Element)node).getTagName().equals(GUIKitUtils.ELEM_ARGS)) {
        Element element1 = (Element)node;
        // Here we can also check for a ref attribute on the args element and
        // do a lookup for a GUIKitTypedObject[] and add them first to aobj
        // Probably combine id with any children to make aobj
        // TODO this should probably be a scoped lookup for max SCOPE_COMPONENT if we register the arguments there
        String id = GUIKitUtils.getAttribute(element1, GUIKitUtils.ATT_REF);
        if (id != null) {
          try {
            // They may not be able to be OBJECT but COMPONENT
            argId = environ.lookupObject(id + ".typedObject", MathematicaObjectRegistry.SCOPE_WIDGET);
            if (argId == null || !(argId instanceof GUIKitTypedObject[])) {
              Object argIdObject = environ.lookupObject(id, MathematicaObjectRegistry.SCOPE_WIDGET);
              if (argIdObject != null && argIdObject instanceof Object[]) {
                // We need to build a GUIKitTypedObject[] equivalent then
                argId = new GUIKitTypedObject[((Object[])argIdObject).length];
                for (int i = 0; i < ((Object[])argIdObject).length; ++i) {
                  Object obj = ((Object[])argIdObject)[i];
                  if (obj != null) ((GUIKitTypedObject[])argId)[i] = (GUIKitTypedObject)TypedObjectFactory.create(obj);
                  }
                }
              else if (argIdObject != null) {
                if (argIdObject instanceof GUIKitTypedObject)
                  argId = new GUIKitTypedObject[]{(GUIKitTypedObject)argIdObject};
                else 
                  argId = new GUIKitTypedObject[]{
                     (GUIKitTypedObject)TypedObjectFactory.create(argIdObject)};
                }
              else argId = null;
              }
            }
          catch (IllegalArgumentException ie) {}
          if (argId != null && argId instanceof GUIKitTypedObject[]) {
            argLength += ((GUIKitTypedObject[])argId).length;
            }
          }
        }

    if(argLength > 0) {
      argObjects = new Object[argLength];
      argClasses = new Class[argLength];
      int i, counter = 0;
      if (argId != null) {
        GUIKitTypedObject[] argArray = (GUIKitTypedObject[])argId;
        for(i = 0; i < argArray.length; i++) {
          GUIKitTypedObject b = argArray[i];
          argObjects[counter] = b.value;
          argClasses[counter] = b.type;
          counter++;
          }
        }
      for( i = 0; i < childResult.length; i++) {
        GUIKitTypedObject b = childResult[i];
        argObjects[counter] = b.value;
        argClasses[counter] = b.type;
        counter++;
        }
      }

    Class targetClass = targetBean.value.getClass();
    if(targetBean.type == (java.lang.Class.class)) {
      targetClass = (Class)targetBean.value;
      }

    Method method = null;
    try {
      method = MathematicaMethodUtils.getMatchingAccessibleMethod(targetClass, methodName, argClasses);
      if (method == null)
       throw new NoSuchMethodException("No such accessible method: " +
         methodName + "("+ MathematicaMethodUtils.createClassParametersString(argClasses) + ") on target class: " + targetClass.getName());
      }
    catch(Exception exception) {
      throw new GUIKitException(GUIKitException.REASON_GET_METHOD_ERROR, exception.getMessage(), exception);
      }

    InvokeMode mode = GUIKitUtils.determineInvokeMode(
      GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_INVOKETHREAD), 
      GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_INVOKEWAIT) );
    if (environ != null) mode.setManager( environ.getBSFManager());
    GUIKitTypedObject callResultBean = (GUIKitTypedObject)TypedObjectFactory.create( method.getReturnType(),
      GUIKitUtils.invokeMethod(method, targetBean.value, argObjects, argClasses, mode));

    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify this should be default scope
    if(id != null)
      environ.registerObject(id, callResultBean.value);

    return callResultBean;
    }

  protected GUIKitTypedObject processString(GUIKitEnvironment environ, Element element, GUIKitTypedObject bean, URL url) throws GUIKitException {
    GUIKitTypedObject bean1 = null;
    String s = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_VALUE);
    if(s != null)
        bean1 = (GUIKitTypedObject)TypedObjectFactory.create(java.lang.String.class, s);
    else
        bean1 = (GUIKitTypedObject)TypedObjectFactory.create(java.lang.String.class, GUIKitUtils.getChildCharacterData(element));
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify this should be default scope
    if(id != null)
        environ.registerObject(id, bean1.value);
    return bean1;
    }

  protected GUIKitTypedObject processInteger(GUIKitEnvironment environ, Element element, GUIKitTypedObject bean, URL url) throws GUIKitException {
    GUIKitTypedObject bean1 = null;
    String s = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_VALUE);
    if(s != null)
        bean1 = (GUIKitTypedObject)TypedObjectFactory.create(int.class, new Integer(s));
    else
        bean1 = (GUIKitTypedObject)TypedObjectFactory.create(int.class, new Integer(GUIKitUtils.getChildCharacterData(element)));
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify this should be default scope
    if(id != null)
        environ.registerObject(id, bean1.value);
    return bean1;
    }
  
  protected GUIKitTypedObject processDouble(GUIKitEnvironment environ, Element element, GUIKitTypedObject bean, URL url) throws GUIKitException {
    GUIKitTypedObject bean1 = null;
    String s = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_VALUE);
    if(s != null)
        bean1 = (GUIKitTypedObject)TypedObjectFactory.create(double.class, new Double(s));
    else
        bean1 = (GUIKitTypedObject)TypedObjectFactory.create(double.class, new Double(GUIKitUtils.getChildCharacterData(element)));
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify this should be default scope
    if(id != null)
        environ.registerObject(id, bean1.value);
    return bean1;
    }
    
   protected GUIKitTypedObject processNull(GUIKitEnvironment environ, Element element, GUIKitTypedObject bean, URL url) throws GUIKitException {
    GUIKitTypedObject bean1 = null;
    String s = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_CLASS);
    if(s != null) {
      Class classAtt = environ.resolveClassName(s);
      if (classAtt != null) bean1 = (GUIKitTypedObject)TypedObjectFactory.create(classAtt, null);
      else bean1 = GUIKitTypedObject.TYPED_NULL;
      }
    else
        bean1 = GUIKitTypedObject.TYPED_NULL;
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify this should be default scope
    if(id != null)
        environ.registerObject(id, bean1.value);
    return bean1;
    }
    

  protected GUIKitTypedObject processBindEvent(GUIKitEnvironment environ, Element element, 
  		GUIKitTypedObject bean, URL url) throws GUIKitException {
    String target = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_TARGET);
    GUIKitTypedObject targetBean = resolveObjectName(environ, target, bean);
    String eventName = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_NAME).toLowerCase();
    String eventFilter = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_FILTER);
    Node node;
    for(node = element.getFirstChild(); node != null; node = node.getNextSibling())
        if(node.getNodeType() == Node.ELEMENT_NODE)
            break;

    Element firstChildElement = (Element)node;

    InvokeMode mode = GUIKitUtils.determineInvokeMode(
      GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_INVOKETHREAD), 
      GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_INVOKEWAIT) );
    if (environ != null) mode.setManager(environ.getBSFManager());
    GUIKitTypedObject resultBean = null;
    
    // TODO make sure this special case is handled properly everywhere
    if (eventName.equalsIgnoreCase("endModal")) {
      // We will make this call unless the flag noting EndModal script was
      // set by a Run/RunModal call
      if (!externalEndModal)
        resultBean = GUIKitUtils.registerEndModalScript(environ, targetBean, 
          firstChildElement, bean, url, mode);
      }
    else if (firstChildElement.getTagName().equals(GUIKitUtils.ELEM_WIDGET)) {
      GUIKitTypedObject childBean = processComponent(environ, firstChildElement, bean, url, null);
      resultBean = GUIKitUtils.addEventListener(environ, targetBean, eventName, eventFilter, childBean, mode);
      }
    else {
      resultBean = GUIKitUtils.bindEventToElement(environ, targetBean, eventName, eventFilter, 
        firstChildElement, bean, url, mode);
      }
    
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify use of default scope
    if(id != null)
      environ.registerObject(id, resultBean.value);
    return resultBean;
    }

  protected void processExpose(GUIKitEnvironment environ, Element element, Hashtable exposeHash) throws GUIKitException {
    String ref = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_REF);
    String as = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_AS);

    if (ref != null && exposeHash != null) {
      if (as != null) exposeHash.put(ref, as);
      else exposeHash.put(ref, ref);
      }
    }

  private GUIKitTypedObject processComponentSource(GUIKitEnvironment environ, Element element, 
         GUIKitTypedObject parentComponent, URL url, String src, GUIKitLayoutInfo parentLayoutInfo)  throws GUIKitException {
      GUIKitTypedObject args[] = null;
      Hashtable exposeHash = new Hashtable();
      Node node = null;

      for(node = element.getFirstChild(); node != null; node = node.getNextSibling())
          if(node.getNodeType() == Node.ELEMENT_NODE &&
            ((Element)node).getTagName().equals(GUIKitUtils.ELEM_EXPOSE)) {
          processExpose(environ, (Element)node, exposeHash);
          }

      node = null;

      for(node = element.getFirstChild(); node != null; node = node.getNextSibling())
          if(node.getNodeType() == Node.ELEMENT_NODE) break;

      if(node != null && ((Element)node).getTagName().equals(GUIKitUtils.ELEM_ARGS)) {
        Element element1 = (Element)node;
        GUIKitTypedObject[] childResult = null;
        try {
          childResult = processChildren(environ, element1, parentComponent, url, parentLayoutInfo, 
              ARGGROUP_RESULTMASK | DEFAULT_RESULTMASK);
          }
        catch(NullPointerException np) {
          if(parentComponent == null)
            throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "cannot refer to context component in outermost <component> element");
          else
            throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, "got null pointer exception: " + np);
          }

        int argLength = 0;

        if (childResult != null)
          argLength += childResult.length;

        // Here we can also check for a ref attribute on the args element and
        // do a lookup for a GUIKitTypedObject[] and add them first to aobj
        // Probably combine id with any children to make aobj
        // TODO this should probably be a scoped lookup for max SCOPE_COMPONENT if we register the arguments there
        Object argId = null;
        String id = GUIKitUtils.getAttribute(element1, GUIKitUtils.ATT_REF);
        if (id != null) {
          try {
            // They may not be able to be OBJECT but COMPONENT
            argId = environ.lookupObject(id + ".typedObject", MathematicaObjectRegistry.SCOPE_WIDGET);
            if (argId == null || !(argId instanceof GUIKitTypedObject[])) {
              Object argIdObject = environ.lookupObject(id, MathematicaObjectRegistry.SCOPE_WIDGET);
              if (argIdObject != null && argIdObject instanceof Object[]) {
                // We need to build a GUIKitTypedObject[] equivalent then
                argId = new GUIKitTypedObject[((Object[])argIdObject).length];
                for (int i = 0; i < ((Object[])argIdObject).length; ++i) {
                  Object obj = ((Object[])argIdObject)[i];
                  if (obj != null) ((GUIKitTypedObject[])argId)[i] = (GUIKitTypedObject)TypedObjectFactory.create(obj);
                  }
                }
              else if (argIdObject != null) {
                if (argIdObject instanceof GUIKitTypedObject)
                  argId = new GUIKitTypedObject[]{(GUIKitTypedObject)argIdObject};
                else 
                  argId = new GUIKitTypedObject[]{
                     (GUIKitTypedObject)TypedObjectFactory.create(argIdObject)};
                }
              else argId = null;
              }
            }
          catch (IllegalArgumentException ie) {}
          if (argId != null && argId instanceof GUIKitTypedObject[]) {
            argLength += ((GUIKitTypedObject[])argId).length;
            }
          }
        if(argLength > 0) {
          args = new GUIKitTypedObject[argLength];
          int i, counter = 0;
          if (argId != null) {
            GUIKitTypedObject[] argArray = (GUIKitTypedObject[])argId;
            for(i = 0; i < argArray.length; i++) {
              GUIKitTypedObject b = argArray[i];
              args[counter] = b;
              counter++;
              }
            }
          for( i = 0; i < childResult.length; i++) {
            GUIKitTypedObject b = childResult[i];
            args[counter] = b;
            counter++;
            }
          }
        }
    return GUIKitUtils.createBean(environ, parentComponent, url, src, args, exposeHash, parentLayoutInfo);
    }

  public GUIKitTypedObject processScriptBean(GUIKitEnvironment environ, Element element, GUIKitTypedObject bean, URL url) throws GUIKitException {
    GUIKitTypedObject result = null;
    Document document = null;
    
    try {
      document = parse(url);
      }
    catch (Exception e) {
      throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, e.getMessage());
      }
      
    if (document != null) {
      try {
        result = environ.getDriver().processDocument(environ, document, bean, url, null);
        }
      catch(Throwable throwable) {
        throw new GUIKitException(GUIKitException.REASON_CREATE_OBJECT_ERROR, "Cannot instantiate " + url + ": " + throwable, throwable);
        }
      }
    processChildren(environ, element, result, url, null, NO_RESULTMASK);
    return result;
    }

  private int[] componentChildDimensions(Element element) {
    int count = 0, count2 = 0;
    Node node = null;
    for(node = element.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node.getNodeType() == Node.ELEMENT_NODE) {
        String tagName = ((Element)node).getTagName();
        if (tagName.equals(GUIKitUtils.ELEM_WIDGET) ||
            tagName.equals(GUIKitUtils.ELEM_GROUP) ||
            tagName.equals(GUIKitUtils.ELEM_SPACE) ||
            tagName.equals(GUIKitUtils.ELEM_FILL) ||
            tagName.equals(GUIKitUtils.ELEM_ALIGN)) {
          if (count2 == 0) {
            count2 = componentChildCount((Element)node);
            }
          count++;
          }
        }
      }
    return new int[] {count, count2};  
    }
    
  private int componentChildCount(Element element) {
    int count = 0;
    Node node = null;
    for(node = element.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node.getNodeType() == Node.ELEMENT_NODE) {
        String tagName = ((Element)node).getTagName();
        if (tagName.equals(GUIKitUtils.ELEM_WIDGET) ||
            tagName.equals(GUIKitUtils.ELEM_GROUP) ||
            tagName.equals(GUIKitUtils.ELEM_SPACE) ||
            tagName.equals(GUIKitUtils.ELEM_FILL) ||
            tagName.equals(GUIKitUtils.ELEM_ALIGN)) {
          count++;
          }
        }
      }
    return count;
    }
  
	protected void processSpace(GUIKitEnvironment environ, Element spaceElement, GUIKitTypedObject bean, URL url)  throws GUIKitException {
		if (bean.getLayout() == null) return;
    String valueAtt = GUIKitUtils.getAttribute(spaceElement, GUIKitUtils.ATT_VALUE);
		if (valueAtt != null) {
			bean.getLayout().addSpace( Integer.parseInt(valueAtt));
			}
		}
		
	protected void processFill(GUIKitEnvironment environ, Element fillElement, GUIKitTypedObject bean, URL url)  throws GUIKitException {
		if (bean.getLayout() == null) return;
    bean.getLayout().addFill();
		}
		
	protected void processAlign(GUIKitEnvironment environ, Element alignElement, GUIKitTypedObject bean, URL url)  throws GUIKitException {
		if (bean.getLayout() == null) return;
    String refAtt = GUIKitUtils.getAttribute(alignElement, GUIKitUtils.ATT_REF);
		String fromAtt = GUIKitUtils.getAttribute(alignElement, GUIKitUtils.ATT_FROM);
		String toAtt = GUIKitUtils.getAttribute(alignElement, GUIKitUtils.ATT_TO);
		if (refAtt != null && fromAtt != null && toAtt != null) {
			// add Align object with ref and from+to
			// iff component reference exists and is instance of component
			GUIKitTypedObject refObj = resolveObjectName(environ, refAtt, bean);
			if (refObj != null && refObj.value != null && refObj.value instanceof Component) {
        bean.getLayout().addAlign((Component)refObj.value,
           fromAtt.equals("Before") ? GUIKitLayout.ALIGN_BEFORE : GUIKitLayout.ALIGN_AFTER,
           toAtt.equals("After") ? GUIKitLayout.ALIGN_AFTER : GUIKitLayout.ALIGN_BEFORE);
			  }
			}
    // add plain Align object
		else {
      bean.getLayout().addAlign();
		  }
		  
		}
		
  protected GUIKitTypedObject processComponent(GUIKitEnvironment environ, 
        Element element, GUIKitTypedObject parentComponent, URL url, GUIKitLayoutInfo parentLayoutInfo)  throws GUIKitException {
    String src = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_SRC);

    if (src == null) {
      src = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_CLASS);
      if (src != null) {
        if (!src.startsWith("class:")) src = "class:" + src;
        }
      }

    String ref = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_REF);
    
    GUIKitLayoutInfo layoutInfo = new GUIKitLayoutInfo();
    layoutInfo.setParent(parentLayoutInfo);
     
    String layoutAtt = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_LAYOUT);
    Node node = null;
    Element layoutNode = null;
    for(node = element.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node.getNodeType() == Node.ELEMENT_NODE && 
        ((Element)node).getTagName().equals(GUIKitUtils.ELEM_LAYOUT)) {
        layoutNode = (Element)node;
        break;
        }
      }
    if (layoutNode != null) {
      // It should be ok to pass a null bean parent for this
      processLayoutInfo(environ, layoutNode, null, url, layoutInfo);
      }
    else if (layoutAtt != null) {
      if (layoutAtt.equals("None")) {
        layoutInfo.setGroupType(GUIKitLayout.GROUP_NONE);
        }
      else {
        if (layoutAtt.equals("Column")) layoutInfo.setGroupType(GUIKitLayout.GROUP_COLUMN);
        // Make sure this doesn't get defaulted in XML if user makes no choice in exprs
        else if (layoutAtt.equals("Automatic")) layoutInfo.setGroupType(GUIKitLayout.GROUP_AUTOMATIC);
        else if (layoutAtt.equals("Row")) layoutInfo.setGroupType(GUIKitLayout.GROUP_ROW);
        else if (layoutAtt.equals("Split")) layoutInfo.setGroupType(GUIKitLayout.GROUP_SPLIT);
        else if (layoutAtt.equals("Grid")) {
           layoutInfo.setGroupType(GUIKitLayout.GROUP_GRID);
           layoutInfo.setGroupDimensions(componentChildDimensions(element));
           }
        }
      }
      
    GUIKitTypedObject thisComponent;
            
    // We may need to preprocess any layoutInfo settings before
    // we might call processComponentSource since it will want
    // to add to content in the layout
    boolean withinTabSplit = false;
    boolean needsPopLayout = false;
    boolean needsApplyLayout = false;
      
    if (layoutInfo != null && layoutInfo.getParent() != null && 
      (layoutInfo.getParent().getGroupType() == GUIKitLayout.GROUP_TAB || 
      layoutInfo.getParent().getGroupType() == GUIKitLayout.GROUP_SPLIT))
        withinTabSplit = true;
           
    if (withinTabSplit) {
      GUIKitLayoutInfo localLayout = new GUIKitLayoutInfo();
      localLayout.setParent(layoutInfo);
      
      if (layoutInfo.getParent().getGroupType() == GUIKitLayout.GROUP_TAB)
        parentComponent.getLayout().nestBoxAsTab(localLayout);
      else 
        parentComponent.getLayout().nestBoxAsSplit(localLayout);
      
      needsApplyLayout = true;
      needsPopLayout = true;
      }
                             
    if(ref != null) {
      thisComponent = resolveObjectName(environ, ref, parentComponent);
      if(src != null && src.startsWith("class:")) {
        Class class1 = environ.resolveClassName(src.substring(6, src.length()));
        if(class1 != thisComponent.type)
          throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, "given class '" +
           class1 + "' does not " + "match with actual class '" + thisComponent.type + "' in <component> element");
        }
      }
    else {
      thisComponent = processComponentSource(environ, element, parentComponent, url, src, layoutInfo);
      if (thisComponent != null && thisComponent.value != null && thisComponent.value instanceof GUIKitInitializable) {
        ((GUIKitInitializable)thisComponent.value).guiInit(environ);
        }
      }
       
    // This will currently only be queried and used in
    // handledCustomAddComponent cases when checking group type
    // where this component is added dynamically at some later point
    // and we want to use its layout properties
    if (thisComponent != null) {
      // TODO we only set this info if it is null so we don't reset more than once
      // when going to processComponent in a nested call, this seems to currently
      // be useful so we don't keep resetting the layoutInfo, but is the first
      // call the one to keep?
      if (thisComponent.getLayoutInfo() == null)
        thisComponent.setLayoutInfo(layoutInfo);
      }
      
    // Now that we have a populated layoutInfo, setup state and new layout
    // and properties of this component before visiting children
    
    if (layoutInfo.getBorder() != null) {
      // We have a border on a specific component this supports
      // the setting of a border on a JComponent itself which may have
      // some odd looks if not just done on a JPanel or on a component which already
      // has a border
      if (thisComponent != null && thisComponent.value != null && thisComponent.value instanceof JComponent) {
        ((JComponent)thisComponent.value).setBorder(layoutInfo.getBorder());
        }
      
      }
    
    if (layoutInfo.getGroupType() == GUIKitLayout.GROUP_NONE || 
        (layoutInfo.getGroupType() == GUIKitLayout.GROUP_UNKNOWN && layoutInfo.getParent() != null && 
            layoutInfo.getParent().getGroupType() == GUIKitLayout.GROUP_NONE)) {
        // we have been given a layout object instance, need
        // to set this as the layout manager for this component or group
      if (layoutInfo.getGroupingObject() != null && thisComponent != null && 
          thisComponent.value != null && thisComponent.value instanceof Container &&
				layoutInfo.getGroupingObject() instanceof LayoutManager) {
				Container c = (Container)thisComponent.value;
				if (c instanceof RootPaneContainer) ((RootPaneContainer)c).getContentPane().setLayout((LayoutManager)layoutInfo.getGroupingObject());
				else c.setLayout((LayoutManager)layoutInfo.getGroupingObject());
        }  
      }
    else {
  
      if (parentComponent != null && thisComponent != null) {
      
        // We have a Window component inside the hierarchy this component needs and wants to have its own
        // new layout system for its children and to have them added to this components own layout settings
        if (thisComponent.value != null && thisComponent.value instanceof Window) {
          // do nothing here but after visiting children we should apply layout
          needsApplyLayout = true;
          }
        else if (!handledCustomAddComponent(parentComponent, thisComponent)) {

          if (parentComponent.getLayout() == null) {
            parentComponent.createLayout(new GUIKitLayoutInfo());
            }
          else // ? if (!withinTabSplit) 
            thisComponent.setLayout( parentComponent.getLayout());
           
          if (layoutInfo.getSpacing() >= 0)  
            parentComponent.getLayout().setInterComponentSpacing(layoutInfo.getSpacing());
                  
          // Check for alignment settings before add
          if (layoutInfo.getSecondaryAlignType() != GUIKitLayout.ALIGN_UNKNOWN)
            parentComponent.getLayout().setAlignment(layoutInfo.getPrimaryAlignType(), layoutInfo.getSecondaryAlignType());
          else if (layoutInfo.getPrimaryAlignType() != GUIKitLayout.ALIGN_UNKNOWN)
            parentComponent.getLayout().setAlignment(layoutInfo.getPrimaryAlignType());
              
          if (thisComponent.value != null && thisComponent.value instanceof Component) {
            
            if (!withinTabSplit){
              // Just added, seems ok, test with wizards
              // should we actually be checking if parentComponent is an instance of
              // Container and thus also a ui class that should do an add 
              // to possibly pass this step preventing child ui to jump into a super-parent ui
              if (parentComponent.value != null && parentComponent.value instanceof Container) {
                if (layoutInfo.getStretchingX() == GUIKitLayout.STRETCH_UNKNOWN ||
                    layoutInfo.getStretchingY() == GUIKitLayout.STRETCH_UNKNOWN)
                  parentComponent.getLayout().add((Component)thisComponent.value);
                else 
                  parentComponent.getLayout().add((Component)thisComponent.value, layoutInfo.getStretchingX(), layoutInfo.getStretchingY());
                }
                
              }
            
            }
            
          }
  
        }

      }
        
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    // TODO verify use of default scope
    if (id != null) 
      environ.registerObject(id, thisComponent.value);

    processChildren(environ, element, thisComponent, url, layoutInfo, NO_RESULTMASK);

    if (needsApplyLayout && thisComponent.getLayout() != null) {
      thisComponent.getLayout().applyLayout(thisComponent);
      }
      
    if (thisComponent != null && thisComponent.value != null && 
        thisComponent.value instanceof Window && thisComponent.getLayout() != null) {
      // since this isn't the root container we pack as a convenience here
      // which I think is ok
      if( ((Window)thisComponent.value).getLayout() != null) {
        Runnable r = new WindowPacker((Window)thisComponent.value);
        SwingUtilities.invokeLater(r);
        }
      
      }
    
    if (needsPopLayout)
      parentComponent.getLayout().popLayout();
      
    return thisComponent;
    }

  // This is called to handle special case component layouts that will not use
  // the normal layout system since the parent-child relationship does not
  // use standard LayoutManager APIs
  protected boolean handledCustomAddComponent(GUIKitTypedObject parentComponent, GUIKitTypedObject childComponent) {
    boolean handled = false;
    if (parentComponent == null || parentComponent.value == null || 
        childComponent == null || childComponent.value == null)
      return false;
      
    if (childComponent.getLayoutInfo().getGroupType() == GUIKitLayout.GROUP_NONE) return true;
    
    if (parentComponent.value instanceof JToolBar && childComponent.value instanceof Component) {
      if (childComponent.value instanceof JSeparator)
        ((JToolBar)parentComponent.value).addSeparator();
      else ((JToolBar)parentComponent.value).add((Component)childComponent.value);
      handled = true;
      }
		else if (childComponent.value instanceof JToolBar && parentComponent.value instanceof Container &&
        !parentComponent.getAddedToolbar()) {
			if (parentComponent.value instanceof RootPaneContainer) {
				if (((RootPaneContainer)parentComponent.value).getContentPane() != null) {
					if (((RootPaneContainer)parentComponent.value).getContentPane().getLayout() instanceof BorderLayout) {
						((RootPaneContainer)parentComponent.value).getContentPane().add((JToolBar)childComponent.value, BorderLayout.NORTH);
						parentComponent.setAddedToolbar(true);
            handled = true;
						}
					}
				}
			else {
				if (((Container)parentComponent.value).getLayout() instanceof BorderLayout) {
					((Container)parentComponent.value).add((JToolBar)childComponent.value, BorderLayout.NORTH);
          parentComponent.setAddedToolbar(true);
					handled = true;
				  }
				}
			}
    else if (parentComponent.value instanceof JMenuBar && childComponent.value instanceof JMenu) {
      ((JMenuBar)parentComponent.value).add((JMenu)childComponent.value);
      handled = true;
      }
    else if (parentComponent.value instanceof JMenu) {
      if (childComponent.value instanceof JSeparator) {
        ((JMenu)parentComponent.value).addSeparator();
        handled = true;
        }
      else if (childComponent.value instanceof JMenuItem) {
        ((JMenu)parentComponent.value).add((JMenuItem)childComponent.value);
        handled = true;
        }
      else if (childComponent.value instanceof String) {
        ((JMenu)parentComponent.value).add((String)childComponent.value);
        handled = true;
        }
      else if (childComponent.value instanceof Action) {
        ((JMenu)parentComponent.value).add((Action)childComponent.value);
        handled = true;
        }
      else if (childComponent.value instanceof Component) {
        ((JMenu)parentComponent.value).add((Component)childComponent.value);
        handled = true;
        }
      }
    else if (parentComponent.value instanceof JPopupMenu) {
      if (childComponent.value instanceof JSeparator) {
        ((JPopupMenu)parentComponent.value).addSeparator();
        handled = true;
        }
      else if (childComponent.value instanceof JMenuItem) {
        ((JPopupMenu)parentComponent.value).add((JMenuItem)childComponent.value);
        handled = true;
        }
      else if (childComponent.value instanceof String) {
        ((JPopupMenu)parentComponent.value).add((String)childComponent.value);
        handled = true;
        }
      else if (childComponent.value instanceof Action) {
        ((JPopupMenu)parentComponent.value).add((Action)childComponent.value);
        handled = true;
        }
      }
    else if (parentComponent.value instanceof ButtonGroup && childComponent.value instanceof AbstractButton) {
      ((ButtonGroup)parentComponent.value).add((AbstractButton)childComponent.value);
      handled = true;
      }
    if (handled) {
      childComponent.getLayoutInfo().setGroupType(GUIKitLayout.GROUP_NONE);
      }
    return handled;
    }
    
  protected GUIKitTypedObject processGroup(GUIKitEnvironment environ, 
        Element element, GUIKitTypedObject parentComponent, URL url, 
        GUIKitLayoutInfo parentLayoutInfo, int resultMask)  throws GUIKitException {
     
    GUIKitLayoutInfo layoutInfo = new GUIKitLayoutInfo();
    // unlike processComponent group defaults to Automatic instead of unknown
		layoutInfo.setGroupType(GUIKitLayout.GROUP_AUTOMATIC);
		layoutInfo.setParent(parentLayoutInfo);
    
    String layoutAtt = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_LAYOUT);
    Node node = null;
    Element layoutNode = null;
    for(node = element.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node.getNodeType() == Node.ELEMENT_NODE && 
        ((Element)node).getTagName().equals(GUIKitUtils.ELEM_LAYOUT)) {
        layoutNode = (Element)node;
        break;
        }
      }
    
    boolean shouldPopLayout = false;
    
    if (layoutNode != null) {
      processLayoutInfo(environ, layoutNode, parentComponent, url, layoutInfo);
      }
    else if (layoutAtt != null) {
      if (layoutAtt.equals("None")) {
        layoutInfo.setGroupType(GUIKitLayout.GROUP_NONE);
        }
      else {
        if (layoutAtt.equals("Column")) layoutInfo.setGroupType(GUIKitLayout.GROUP_COLUMN);
        // Make sure this doesn't get defaulted in XML if user makes no choice in exprs
        else if (layoutAtt.equals("Automatic")) layoutInfo.setGroupType(GUIKitLayout.GROUP_AUTOMATIC);
        else if (layoutAtt.equals("Row")) layoutInfo.setGroupType(GUIKitLayout.GROUP_ROW);
        else if (layoutAtt.equals("Split")) layoutInfo.setGroupType(GUIKitLayout.GROUP_SPLIT);
        else if (layoutAtt.equals("Grid")) {
           layoutInfo.setGroupType(GUIKitLayout.GROUP_GRID);
           layoutInfo.setGroupDimensions(componentChildDimensions(element));
           }
        }
      }
    
    if (layoutInfo.getGroupType() == GUIKitLayout.GROUP_NONE ||
        (layoutInfo.getGroupType() == GUIKitLayout.GROUP_UNKNOWN && layoutInfo.getParent() != null && 
            layoutInfo.getParent().getGroupType() == GUIKitLayout.GROUP_NONE)
        ) {
     // we have been given a layout object instance, need
	   // to set this as the layout manager for this component or group
     if (layoutInfo.getGroupingObject() != null && parentComponent != null && 
          parentComponent.value != null && parentComponent.value instanceof Container &&
	       layoutInfo.getGroupingObject() instanceof LayoutManager) {
					Container c = (Container)parentComponent.value;
					if (c instanceof RootPaneContainer) ((RootPaneContainer)c).getContentPane().setLayout((LayoutManager)layoutInfo.getGroupingObject());
					else c.setLayout((LayoutManager)layoutInfo.getGroupingObject());
	       }  
      }
    else {
      
      if (parentComponent != null) { 
        if (parentComponent.getLayout() == null) {
          parentComponent.createLayout(new GUIKitLayoutInfo());
          } 
        
        // If this wants to be the root group it should call createLayout and not push
        String isRoot = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ROOT);
        boolean isRootGroup = (isRoot != null && isRoot.equalsIgnoreCase("true"));
        if (isRootGroup) parentComponent.createLayout(layoutInfo);
        else if (layoutInfo.getGroupType() != GUIKitLayout.GROUP_UNKNOWN)
          shouldPopLayout = parentComponent.getLayout().pushLayout(layoutInfo);
        
        // Will this ever be valid for a component walk or only for a group?
        if (layoutInfo.getBorder() != null)
          parentComponent.getLayout().addBorder(layoutInfo.getBorder());
            
        if (layoutInfo.getSpacing() >= 0)  
          parentComponent.getLayout().setInterComponentSpacing(layoutInfo.getSpacing());
              
        // Check for alignment settings before any adds
        if (layoutInfo.getSecondaryAlignType() != GUIKitLayout.ALIGN_UNKNOWN)
          parentComponent.getLayout().setAlignment(layoutInfo.getPrimaryAlignType(), layoutInfo.getSecondaryAlignType());
        else if (layoutInfo.getPrimaryAlignType() != GUIKitLayout.ALIGN_UNKNOWN)
          parentComponent.getLayout().setAlignment(layoutInfo.getPrimaryAlignType());
            
				// No stretching checks since we are not adding any components right now
        }
      }
    
    String id = GUIKitUtils.getAttribute(element, GUIKitUtils.ATT_ID);
    if (id != null && parentComponent != null && parentComponent.getLayout() != null && 
      parentComponent.getLayout().getCurrentContainer() != null) {
        environ.registerObject(id, parentComponent.getLayout().getCurrentContainer());
      }
      
    GUIKitTypedObject[] result = null;
    result = processChildren(environ, element, parentComponent, url, layoutInfo, resultMask);
    
    if (parentComponent != null && parentComponent.getLayout() != null) {
      // This was recently added to allow things like a Panel to apply its
      // children content even if it is not the root panel.
      // This seems to be safe to add after testing all other examples
      if (parentComponent.isRootComponent()) {
        parentComponent.getLayout().applyLayout(parentComponent);
        }
      
      if (shouldPopLayout) {
        parentComponent.getLayout().popLayout();   
        }
      }
    
    // group does not currently create any new beans for the hierarchy unless
    // result Mask is appropriate for certain situations (like arg groups)
    if ((resultMask & ARGGROUP_RESULTMASK) != 0 && result != null) {
      boolean twoDimArray = true;
      int arrayCount = -1;
      for (int i = 0 ; i < result.length; ++i) {
        if (result[i] != null && result[i].type == Object[].class && result[i].value != null) {
          if (arrayCount < 0 || arrayCount == ((Object[])result[i].value).length) {
            arrayCount = ((Object[])result[i].value).length;
            }
          else {twoDimArray = false; break;}
          }
        else {twoDimArray = false; break;}
        }
      if (twoDimArray) {
        Object[][] objs = new Object[result.length][];
        for (int i = 0 ; i < result.length; ++i) {
          objs[i] = (result[i] != null ? ((Object[])result[i].value) : null);
          }
        return (GUIKitTypedObject)TypedObjectFactory.create(Object[][].class, objs);
        }
      else {
        Object[] objs = new Object[result.length];
        for (int i = 0 ; i < result.length; ++i) {
          objs[i] = (result[i] != null ? result[i].value : null);
          }
        return (GUIKitTypedObject)TypedObjectFactory.create(Object[].class, objs);
        }
      }
    else return null;
    }
    
  protected void processLayoutTabs(GUIKitEnvironment environ, Element groupElement, 
    GUIKitTypedObject bean, URL url, GUIKitLayoutInfo layoutInfo)  throws GUIKitException {
    Node node = null;
    for(node = groupElement.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node.getNodeType() == Node.ELEMENT_NODE) {
        Element elem = (Element)node;
        String tagName = elem.getTagName();
        if(tagName.equals(GUIKitUtils.ELEM_TABS)) {
          String orientAtt = GUIKitUtils.getAttribute(elem, GUIKitUtils.ATT_ORIENT);
          if (orientAtt != null) {
            if (orientAtt.equals("Left")) {
              layoutInfo.setTabPlacement(GUIKitLayout.TABPLACEMENT_LEFT);
              }
            else if (orientAtt.equals("Right")) {
              layoutInfo.setTabPlacement(GUIKitLayout.TABPLACEMENT_RIGHT);
              }
            else if (orientAtt.equals("Bottom")) {
              layoutInfo.setTabPlacement(GUIKitLayout.TABPLACEMENT_BOTTOM);
              }
            else
              layoutInfo.setTabPlacement(GUIKitLayout.TABPLACEMENT_TOP);
            }
          else layoutInfo.setTabPlacement(GUIKitLayout.TABPLACEMENT_TOP);
          
          // Try to read any string names
          Vector tabStrings = new Vector();
          Node childNode = null;
          for(childNode = elem.getFirstChild(); childNode != null; childNode = childNode.getNextSibling()) {
            if(childNode.getNodeType() == Node.ELEMENT_NODE) {
              Element childElem = (Element)childNode;
              String childTagName = childElem.getTagName();
              if(childTagName.equals(GUIKitUtils.ELEM_STRING)) {
                GUIKitTypedObject oneString = processString(environ, childElem, bean, url);
                if (oneString != null && oneString.value != null)
                  tabStrings.add(oneString.value);
                }
              }
            }
          
          layoutInfo.setTabNames( (String[])tabStrings.toArray(new String[]{}) );
          
          }
        }
      }
    }
    
  protected void processLayoutSplit(GUIKitEnvironment environ, Element groupElement, 
    GUIKitTypedObject bean, URL url, GUIKitLayoutInfo layoutInfo)  throws GUIKitException {
    Node node = null;
    for(node = groupElement.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node.getNodeType() == Node.ELEMENT_NODE) {
        Element elem = (Element)node;
        String tagName = elem.getTagName();
        if(tagName.equals(GUIKitUtils.ELEM_SPLIT)) {
          String orientAtt = GUIKitUtils.getAttribute(elem, GUIKitUtils.ATT_ORIENT);
          if (orientAtt != null) {
            if (orientAtt.equals("Horizontal")) {
              layoutInfo.setSplitOrientation(GUIKitLayout.SPLIT_HORIZONTAL);
              }
            else
              layoutInfo.setSplitOrientation(GUIKitLayout.SPLIT_VERTICAL);
            }
          else layoutInfo.setSplitOrientation(GUIKitLayout.SPLIT_VERTICAL);
          }
        }
      }
    }
    
  protected void processLayoutInfo(GUIKitEnvironment environ, Element layoutElement, 
        GUIKitTypedObject bean, URL url, GUIKitLayoutInfo layoutInfo)  throws GUIKitException {
    Node node = null;
    for(node = layoutElement.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node.getNodeType() == Node.ELEMENT_NODE) {
        Element elem = (Element)node;
        String tagName = elem.getTagName();
        String typeAtt = GUIKitUtils.getAttribute(elem, GUIKitUtils.ATT_TYPE);
        
        if(tagName.equals(GUIKitUtils.ELEM_GROUPING)) {
          if (typeAtt != null) {
            if (typeAtt.equals("None")) {
              layoutInfo.setGroupType(GUIKitLayout.GROUP_NONE);
              }
            else {
              if (typeAtt.equals("Column")) layoutInfo.setGroupType(GUIKitLayout.GROUP_COLUMN);
              else if (typeAtt.equals("Automatic")) layoutInfo.setGroupType(GUIKitLayout.GROUP_AUTOMATIC);
              else if (typeAtt.equals("Row")) layoutInfo.setGroupType(GUIKitLayout.GROUP_ROW);
              else if (typeAtt.equals("Split")) {
                layoutInfo.setGroupType(GUIKitLayout.GROUP_SPLIT);
                // Need to walk split child for orientation
                processLayoutSplit(environ, elem, bean, url, layoutInfo);
                }
              else if (typeAtt.equals("Tabs")) {
                layoutInfo.setGroupType(GUIKitLayout.GROUP_TAB);
                // Need to walk tabs child for orientation and tab strings
                processLayoutTabs(environ, elem, bean, url, layoutInfo);
                }
              else if (typeAtt.equals("Grid")) {
                layoutInfo.setGroupType(GUIKitLayout.GROUP_GRID);
                layoutInfo.setGroupDimensions(componentChildDimensions((Element)elem.getParentNode().getParentNode()));
                }
              }
            }
          // We check for grouping specified as an explicit layout object
          Element componentNode = null;
          Node compNode = null;
          for(compNode = elem.getFirstChild(); compNode != null; compNode = compNode.getNextSibling()) {
            if(compNode.getNodeType() == Node.ELEMENT_NODE && 
                  ((Element)compNode).getTagName().equals(GUIKitUtils.ELEM_WIDGET)) {
              componentNode = (Element)compNode;
              break;
              }
            }
          if (componentNode != null) {
            GUIKitTypedObject groupingObject = processComponent(environ, componentNode, bean, url, null);
            if (groupingObject != null && groupingObject.value != null &&
                groupingObject.value instanceof LayoutManager) {
              // Grouping has specified a layout manager with manual handling of children then
              // Either we allow setting of a layout manager or instead of this obj, provide a
              // "Manual" choice and assume user sets the layout property and calls add themselves somewhere
              layoutInfo.setGroupingObject(groupingObject);
              }
            }
      
      
          }
          
        else if(tagName.equals(GUIKitUtils.ELEM_BORDER)) {
          layoutInfo.setBorder(processLayoutBorder(environ, elem, bean, url)); 
          }
          
        else if(tagName.equals(GUIKitUtils.ELEM_ALIGNMENT)) {
          if (typeAtt != null) {
            if (typeAtt.indexOf(',') > 0) {
              // Automatic, Left, Center, Right
              String alignOne = typeAtt.substring(0, typeAtt.indexOf(',')).trim();
              if (alignOne.equals("Left")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_LEFT);
              else if (alignOne.equals("Center")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_CENTER);
              else if (alignOne.equals("Right")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_RIGHT);
              else if (alignOne.equals("Automatic")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_AUTOMATIC);
              // Automatic, Top, Center, Bottom
              String alignTwo = typeAtt.substring(typeAtt.indexOf(',')+1).trim();
              if (alignTwo.equals("Top")) layoutInfo.setSecondaryAlignType(GUIKitLayout.ALIGN_TOP);
              else if (alignTwo.equals("Center")) layoutInfo.setSecondaryAlignType(GUIKitLayout.ALIGN_CENTER);
              else if (alignTwo.equals("Bottom")) layoutInfo.setSecondaryAlignType(GUIKitLayout.ALIGN_BOTTOM);
              else if (alignTwo.equals("Automatic")) layoutInfo.setSecondaryAlignType(GUIKitLayout.ALIGN_AUTOMATIC);
              }
            else {
              // Automatic, Left, Center, Right, Top, Bottom
              typeAtt = typeAtt.trim();
              if (typeAtt.equals("Left")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_LEFT);
              else if (typeAtt.equals("Center")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_CENTER);
              else if (typeAtt.equals("Right")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_RIGHT);
              else if (typeAtt.equals("Top")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_TOP);
              else if (typeAtt.equals("Bottom")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_BOTTOM);
              else if (typeAtt.equals("Automatic")) layoutInfo.setPrimaryAlignType(GUIKitLayout.ALIGN_AUTOMATIC);
              }
            } 
         }
         
        else if(tagName.equals(GUIKitUtils.ELEM_STRETCHING)) {
          if (typeAtt != null) {
            if (typeAtt.indexOf(',') > 0) {
              // None, False, ComponentAlignments, True, Maximize
              String stretchOne = typeAtt.substring(0, typeAtt.indexOf(',')).trim();
              if (stretchOne.equals("False") || stretchOne.equals("None")) layoutInfo.setStretchingX(GUIKitLayout.STRETCH_NONE);
              else if (stretchOne.equals("ComponentAlign")) layoutInfo.setStretchingX(GUIKitLayout.STRETCH_COMPONENTALIGN);
              else if (stretchOne.equals("True")) layoutInfo.setStretchingX(GUIKitLayout.STRETCH_TRUE);
              else if (stretchOne.equals("Maximize")) layoutInfo.setStretchingX(GUIKitLayout.STRETCH_MAXIMIZE);
              // None, False, ComponentAlignments, True, Maximize
              String stretchTwo = typeAtt.substring(typeAtt.indexOf(',')+1).trim();
              if (stretchTwo.equals("False") || stretchTwo.equals("None")) layoutInfo.setStretchingY(GUIKitLayout.STRETCH_NONE);
              else if (stretchTwo.equals("ComponentAlign")) layoutInfo.setStretchingY(GUIKitLayout.STRETCH_COMPONENTALIGN);
              else if (stretchTwo.equals("True")) layoutInfo.setStretchingY(GUIKitLayout.STRETCH_TRUE);
              else if (stretchTwo.equals("Maximize")) layoutInfo.setStretchingY(GUIKitLayout.STRETCH_MAXIMIZE);
              }
            else {
              // None, False, ComponentAlignments, True, Maximize
              typeAtt = typeAtt.trim();
              if (typeAtt.equals("False") || typeAtt.equals("None")) {
                layoutInfo.setStretchingX(GUIKitLayout.STRETCH_NONE);
                layoutInfo.setStretchingY(GUIKitLayout.STRETCH_NONE);
                }
              else if (typeAtt.equals("ComponentAlign")) {
                 layoutInfo.setStretchingX(GUIKitLayout.STRETCH_COMPONENTALIGN);
                 layoutInfo.setStretchingY(GUIKitLayout.STRETCH_COMPONENTALIGN);
                 }
              else if (typeAtt.equals("True")) {
                 layoutInfo.setStretchingX( GUIKitLayout.STRETCH_TRUE);
                 layoutInfo.setStretchingY( GUIKitLayout.STRETCH_TRUE);
                 }
              else if (typeAtt.equals("Maximize")) {
                 layoutInfo.setStretchingX(GUIKitLayout.STRETCH_MAXIMIZE);
                 layoutInfo.setStretchingY(GUIKitLayout.STRETCH_MAXIMIZE);
                 }
              }
            }
        }
        
        else if(tagName.equals(GUIKitUtils.ELEM_SPACING)) {
          String valueAtt = GUIKitUtils.getAttribute(elem, GUIKitUtils.ATT_VALUE);
          if (valueAtt != null) {
            layoutInfo.setSpacing(Integer.parseInt(valueAtt));
            }
          }
        
        }
      }
      
    }
    
	protected Border processLayoutBorder(GUIKitEnvironment environ, Element borderElement, GUIKitTypedObject bean, URL url)  throws GUIKitException {
    String typeAtt = GUIKitUtils.getAttribute(borderElement, GUIKitUtils.ATT_TYPE);

		if (typeAtt != null && (typeAtt.equals("None") || typeAtt.equals("Automatic")))
			return null;
			
		String titleAtt = GUIKitUtils.getAttribute(borderElement, GUIKitUtils.ATT_TITLE);
		if (titleAtt != null) 
			return BorderFactory.createTitledBorder(titleAtt);
		
		String leftAtt = GUIKitUtils.getAttribute(borderElement, GUIKitUtils.ATT_LEFT);
		String rightAtt = GUIKitUtils.getAttribute(borderElement, GUIKitUtils.ATT_RIGHT);
		String topAtt = GUIKitUtils.getAttribute(borderElement, GUIKitUtils.ATT_TOP);
		String bottomAtt = GUIKitUtils.getAttribute(borderElement, GUIKitUtils.ATT_BOTTOM);
		if (leftAtt != null && rightAtt != null && topAtt != null && bottomAtt != null)
		  return BorderFactory.createEmptyBorder(Integer.parseInt(topAtt), Integer.parseInt(leftAtt), 
		     Integer.parseInt(bottomAtt), Integer.parseInt(rightAtt));
		
		Node node = null;
	  Element componentNode = null;
		for(node = borderElement.getFirstChild(); node != null; node = node.getNextSibling()) {
			if(node.getNodeType() == Node.ELEMENT_NODE && 
					((Element)node).getTagName().equals(GUIKitUtils.ELEM_WIDGET)) {
				componentNode = (Element)node;
        GUIKitTypedObject borderObject = processComponent(environ, componentNode, bean, url, null);
        if (borderObject != null && borderObject.value != null && borderObject.value instanceof Border)
          return (Border)borderObject.value;
				break;
				}
      else if(node.getNodeType() == Node.ELEMENT_NODE && 
          ((Element)node).getTagName().equals(GUIKitUtils.ELEM_INVOKEMETHOD)) {
        componentNode = (Element)node;
        GUIKitTypedObject borderObject = processInvokeMethod(environ, componentNode, bean, url);
        if (borderObject != null && borderObject.value != null && borderObject.value instanceof Border)
          return (Border)borderObject.value;
        break;
        }
			}

	  Border active = null;
		for(node = borderElement.getFirstChild(); node != null; node = node.getNextSibling()) {
		  if(node.getNodeType() == Node.ELEMENT_NODE && 
			   ((Element)node).getTagName().equals(GUIKitUtils.ELEM_BORDER)) {
			  if (active == null) active = processLayoutBorder(environ, (Element)node, bean, url);
			  else active = BorderFactory.createCompoundBorder(active,
           processLayoutBorder(environ, (Element)node, bean, url));
				}
			}
  	return active;

		}
		
  protected GUIKitTypedObject processScript(GUIKitEnvironment environ, Element scriptElement, GUIKitTypedObject bean, URL url) throws GUIKitException  {
    String lang = GUIKitUtils.getAttribute(scriptElement, GUIKitUtils.ATT_LANGUAGE);
    String srcAtt = GUIKitUtils.getAttribute(scriptElement, GUIKitUtils.ATT_SRC);
    GUIKitTypedObject result = null;

    // Default language is Mathematica as opposed to XML
    if (lang == null) {
      lang = MathematicaBSFManager.MATHEMATICA_LANGUAGE_NAME;
      }

    Element firstArgElement = null;
    Node node;
    for(node = scriptElement.getFirstChild(); node != null; node = node.getNextSibling())
        if(node.getNodeType() == Node.ELEMENT_NODE)
            break;

    if(node != null && ((Element)node).getTagName().equals(GUIKitUtils.ELEM_ARGS))
        firstArgElement = (Element)node;

    // If we have a src attribute need to use this external content for script
    if (srcAtt != null) {
      URL srcURL = null;
      srcURL = MathematicaEngineUtils.getMathematicaURL(GUIKitEnvironment.RESOLVE_FUNCTION,
          url, srcAtt, environ.getMathematicaEngine(), getURLCache());
      if (srcURL == null) {
        // A source attribute was not found we currently consider this a fatal error
        throw new GUIKitException(GUIKitException.REASON_INVALID_ARGUMENT, 
          "Unable to load Script source :" + srcAtt);
        }
      
      if(lang.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_GUI_XMLFORMAT) || 
         lang.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_XML)) {
        // This may not expose subelement ids when in an external file than when explicitly
        // in the document. Need to see if this is an issue
        result = processScriptBean(environ, scriptElement, bean, srcURL);
        }
      else {
        // needs to be contents of file
        Object scriptContent = null;
        // if srcURL is file protocol we evaluate this 
        // script content by passing the file path and evaluating Script with Get
        try {
           if (srcURL != null && "file".equals(srcURL.getProtocol())) {
            scriptContent = srcURL;
            //System.out.println("File Get processScript: " + srcURL.getPath());
            }
          else {
            scriptContent = MathematicaEngineUtils.getContentAsString(srcURL);
            }
          }
        catch (IOException ie) {
          throw new GUIKitException(GUIKitException.REASON_OTHER_ERROR, ie.getMessage());
          }
        if(bean != null) {
          env.pushObjectRegistry(MathematicaObjectRegistry.SCOPE_ACTION, true);
          }
        try {  
          result = GUIKitUtils.evaluateBSFScript(environ, bean, srcURL, firstArgElement, lang, scriptContent);
          }
        finally {
          if(bean != null)
            env.popObjectRegistry(true);
          }
        }
      }
    // if no src attribute content for script is within existing Document
    else {
      if(bean != null) {
        env.pushObjectRegistry(MathematicaObjectRegistry.SCOPE_ACTION, true);
        }
      try {
        if( lang.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_GUI_XMLFORMAT) || 
            lang.equalsIgnoreCase(GUIKitUtils.ATTVAL_LANGUAGE_XML)) {
              
          if(firstArgElement != null) {
            GUIKitTypedObject[] childResults = env.getDriver().processChildren(env, firstArgElement, bean, url, null, DEFAULT_RESULTMASK);
            // Under what cases does this get called, should we be adding a scope input
            // to control whether this is always ACTION??
            GUIKitUtils.registerAsScopeArguments(env, childResults, MathematicaObjectRegistry.SCOPE_ACTION);
            }
          result = GUIKitUtils.evaluateGUIKitElement(environ, bean, url, scriptElement, MathematicaObjectRegistry.SCOPE_ACTION);
          }
        else {
          String scriptContent = GUIKitUtils.getChildCharacterData(scriptElement);
          result = GUIKitUtils.evaluateBSFScript(environ, bean, url, firstArgElement, lang, scriptContent);
          }
        }
      finally {
        if(bean != null)
          env.popObjectRegistry(true);
        }
      }
    return result;
    }

  protected Object processNode(GUIKitEnvironment environ, Node node, URL url) throws GUIKitException {
    GUIKitTypedObject bean = processNode(environ, node, null, url, null);
    if(bean == null)
      return null;
    else
    return bean.value;
    }

  public Object processNode(GUIKitEnvironment environ, Node node) throws GUIKitException {
    return processNode(environ, node, null);
    }

  protected GUIKitTypedObject processDocument(GUIKitEnvironment environ, Document document,
        GUIKitTypedObject parentComponent, URL url, GUIKitLayoutInfo parentLayoutInfo) throws GUIKitException {
    Element element = document.getDocumentElement();
    if(element == null)
      return null;
    GUIKitTypedObject documentComponent = null;
    
    // For purposes of finding relative resources a driver should build
    // a document URL stack and make this available when Mathematica searches on $Path
    if (url != null) {
      documentURLStack.push(url);
      }
      
    for(Node node = document.getFirstChild(); node != null; node = node.getNextSibling()) {
      if(node == element)
        documentComponent = processNode(environ, node, parentComponent, url, parentLayoutInfo);
      else if(node.getNodeType() == Node.PROCESSING_INSTRUCTION_NODE)
        processPI(environ, (ProcessingInstruction)node);
      }
      
    if (url != null) {
      documentURLStack.pop();
      }
  
		if (documentComponent != null &&  documentComponent.getLayout() != null) {
			documentComponent.getLayout().applyLayout(documentComponent);
 			}
          
    return documentComponent;
    }
    
  private class CloseAction extends AbstractAction {
    private static final long serialVersionUID = -1247987675476728948L;
    private Window window;
    public CloseAction(Window win) {
      this.window = win;
      }
    public void actionPerformed(ActionEvent e) {
      requestClose(window);
      }
    public void setWindow(Window win) {this.window = win;}
    }
          
  private static class WindowPacker implements Runnable {
    private Window window;
    public WindowPacker(Window win) {
      this.window = win;
      }
    public void run() {
      window.pack();
      WindowUtils.centerComponentIfNeeded(window);
      window = null;
      }
    }
  
  private class WindowShower implements Runnable {
    private Window window;
    private boolean needsPack;
    private boolean useJavaShow;
    public WindowShower(Window win, boolean javaShow, boolean pack) {
      this.window = win;
      this.useJavaShow = javaShow;
      this.needsPack = pack;
      }
    public void run() {
      if (needsPack) {
        window.pack();
        WindowUtils.centerComponentIfNeeded(window);
        }
      if (useJavaShow) requestJavaShow(window);
      else window.show();
      window = null;
      }
    }
    
  private class ModalInterruptTimerTask extends TimerTask {
    private GUIKitDriver driver;
    private Object executeObject;
    private boolean wasInterrupted = false;
    
    public ModalInterruptTimerTask(GUIKitDriver d, Object e) {
      executeObject = e;
      driver = d;
      }
    public boolean getWasInterrupted() {return wasInterrupted;}
    public void run() {
      if (driver == null) return;
      KernelLink ml = StdLink.getLink();
      if (ml != null && ml.wasInterrupted()) {
        driver.requestClose(executeObject);
        driver = null;
        executeObject = null;
        }
      }
  }
}
