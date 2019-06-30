/*
 * @(#)GUIKitJApplet.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.applet;

import java.awt.BorderLayout;
import java.awt.Frame;

import javax.swing.JApplet;
import javax.swing.JLabel;

import com.wolfram.guikit.GUIKitDriverContainer;

/**
 * GUIKitJApplet is an implementation of the GUIKitDriver as a JApplet
 *
 * Will it ever make sense for this to support/use the modal API, probably not
 */
public class GUIKitJApplet extends JApplet implements GUIKitDriverContainer {

  private static final long serialVersionUID = -1282987975353798948L;
    
  protected Object appletObject = null;
  protected JAppletDriver driver = null;

  // in applet mode driver: assume parameter "source" to describe the resource to run
  public void init() {
    initDriver();
    initGUI();
    }

  protected void initDriver(){
    driver = new JAppletDriver(this);
    boolean debug = false;
    String useDebug = getParameter(GUIKitApplet.PARAM_DEBUG);
    if (useDebug != null && (useDebug.equalsIgnoreCase("true") || useDebug.equalsIgnoreCase("yes")))
      debug = true;
    driver.setDebug(debug);
    }
    
  protected void initGUI() {
     try {
      // First see if raw definition is specified by the expr parameter
      // as a Mathematica expression
      String content = getParameter(GUIKitApplet.PARAM_EXPR);
      if (content != null) {
        appletObject = driver.loadExpressionContent(content);
        return;
        }
        
      // Next see if raw definition is specified by the xml parameter
      // as a GUI_XMLFORMAT definition
      content = getParameter(GUIKitApplet.PARAM_XML);
      if (content != null) {
        appletObject = driver.loadContent(content);
        return;
        }
        
      // Fallback to see if external resource path is specified by src parameter
      content = getParameter(GUIKitApplet.PARAM_SRC);
      if (content != null) {
        appletObject = driver.loadFile(content);
        return;
        }
        
      }
    catch (Exception e) {
      e.printStackTrace();
      }
    }
    
  public void driverFinished() {
    stop();
    }

  public void start() {
    if( appletObject == null) {
      init();
      }
    if ((appletObject != null) && (!driver.getIsRunning())) {
      driver.execute(appletObject, JAppletDriver.RELEASE_ONCLOSE, true);
      }
    else
      getContentPane().add(new JLabel("Applet parameters did not resolve to a valid GUIKit definition."), BorderLayout.CENTER);
    super.start();
    }

  public void stop() {
    // Currently a GUIKit driver does nothing for stop
    // Would certain GUIKit content want to be notified of this though
    // so they could trigger certain events
    super.stop();
    }

  public void destroy() {
    if (appletObject != null && !appletObject.equals(this)) {
      if (appletObject instanceof Frame)
        ((Frame)appletObject).dispose();
      driver.destroy(appletObject);
      }
    super.destroy();
    }

}
