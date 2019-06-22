/*
 * @(#)AppletDriver.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.applet;

import java.applet.Applet;
import java.awt.Component;
import java.awt.Label;
import java.awt.Window;

import javax.swing.SwingUtilities;

import com.wolfram.guikit.*;
import com.wolfram.guikit.util.WindowUtils;
import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.MathematicaObjectRegistry;

/**
 * AppletDriver
 *
 * Will it ever make sense for this to support/use the modal API, probably not
 */
public class AppletDriver extends GUIKitDriver {

  protected GUIKitDriverContainer container;
    
  public AppletDriver(GUIKitDriverContainer applet) {
    super();
    this.container = applet;
    createEnvironment();
    }
  
  protected void initEnvironment() {
    // Applets will first be useful in JavaCells so the default new kernel
    // is chosen to be 6.0. This can be easily overridden
    registerObject(MathematicaBSFEngine.ID_KERNELLINKVERSION, "6.0", MathematicaObjectRegistry.SCOPE_OBJECT);
    
    String customCommandLine = null;
    customCommandLine = ((Applet)container).getParameter(GUIKitApplet.PARAM_KERNELLINKCOMMANDLINE);
    if (customCommandLine != null) {
			setLinkCommandLine(customCommandLine);
      }
    super.initEnvironment();
    }
  
	public Object loadFile(String file) {
		return super.loadFile(file);
		}
		
  public Object loadContent(String content) {
    return super.loadContent(content);
    }
    
  public Object loadExpressionContent(String content) {
    String xmlString = null;
    try {
      xmlString = getGUIKitEnvironment().getMathematicaEngine().requestExprToXMLString(
        content, GUIKitUtils.GUI_XMLFORMAT);
      }
    catch (Exception e) {e.printStackTrace();}
    if (xmlString != null)
      return super.loadContent( xmlString);
    else {
      System.err.println("Unable to produce a valid definition from expression content :\n" + content);
      return null;
      }
    }
    
  // When run in Mathematica or an application, GUIKitDriver in createEnvironment()
  // will set a GUIKit look and feel once in the lifetime of the VM,
  // but applets have their own LAF context so we can set it each time
  protected void createEnvironment() {
    prepareInitialLookAndFeel();
    super.createEnvironment();
    }
  
  public Object resolveExecuteObject(Object sourceObject) {
    if (!(container instanceof Applet)) return null;
    
    Object executeObject = null;
    Applet applet = (Applet)container;
    
    if ((sourceObject instanceof Component)) {
      if (sourceObject instanceof Window) {
        executeObject = (Window)sourceObject;
        applet.add(new Label("GUIKit content is running in separate window"));
        }
      else {
        applet.add((Component)sourceObject);
        executeObject = this;
        }
      }

    return executeObject;
    }

  public void destroyEnvironment(Exception e) {
    super.destroyEnvironment(e);
    if (e != null) {
      e.printStackTrace();
      }
    container.driverFinished();
    container = null;
    }

  public Object execute(final Object executeObject, int releaseMode) {

    if (executeObject != null) {

      if (getGUIKitEnvironment() != null) {
        // These could be combined as one call without registering in the objectRegistry
        // but this could perhaps be useful elsewhere
        registerObject(ID_ROOTOBJECT, executeObject);
        requestDeclareSymbol(ROOTOBJECT_SYMBOL, executeObject);
        }
          
      if (!getIsRunning()) {
        setIsRunning(true);
        setReleaseMode(releaseMode);
        setExecuteAsModal(false);
        
        if (!executeObject.equals(this)) {
          if (executeObject instanceof Window) {
            
            // TODO We might want to issue a special warning here or an error
            // will be printed because of failure?
						if (!requestSharedKernelState()) {
              return null;
              }
            
            ((Window)executeObject).addWindowListener(this);
            
            boolean needsPack = false;
            if (((Window)executeObject).getLayout() != null)
              needsPack = true;
            Runnable r = new WindowShower((Window)executeObject, false, needsPack);
            SwingUtilities.invokeLater(r);
            }
          }
   
        }
      else {
        if (!executeObject.equals(this)) {
          if (executeObject instanceof Window) {
            Runnable r = new WindowShower((Window)executeObject, false, false);
            SwingUtilities.invokeLater(r);
            }
          }
        }
      
      }

    return executeObject;
    }

  public void destroy(Object executeObject) {
    destroyEnvironment();
    }

  protected void handleException(Exception e, MathematicaBSFEngine engine) {
    // By default we will assume the driver/engine is running within Mathematica
    // and only return exceptions as messages
    // Subclasses should override this method if they want to throw the exception in Java
    e.printStackTrace();
    }

  private class WindowShower implements Runnable {
    final Window window;
    final boolean needsPack;
    final boolean useJavaShow;
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
      }
    }
    
}
