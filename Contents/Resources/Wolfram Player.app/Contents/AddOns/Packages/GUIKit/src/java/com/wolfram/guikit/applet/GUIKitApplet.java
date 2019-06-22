/*
 * @(#)GUIKitApplet.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.applet;

import java.awt.*;
import java.applet.*;

import com.wolfram.guikit.GUIKitDriverContainer;

/**
 * GUIKitApplet is an implementation of the GUIKitDriver as an applet
 *
 * Will it ever make sense for this to support/use the modal API, probably not
 */
public class GUIKitApplet extends Applet implements GUIKitDriverContainer {

  private static final long serialVersionUID = -1387987974454748948L;
  
  protected Object appletObject = null;
  protected AppletDriver driver = null;

  public static final String PARAM_SRC = "src";
  public static final String PARAM_EXPR = "expr";
  public static final String PARAM_XML = "xml";
  public static final String PARAM_DEBUG = "debug";
  
	public static final String PARAM_KERNELLINKCOMMANDLINE = "kernelLinkCommandLine";
	
  /* TODO Consider adding:
   * parameter to allow setting of explicit kernel context used "context"
   * paramters for some components that might implement/require arguments
   *   but then how will the string parameters map to argument type calls?
   * 
   */
   
  // in applet mode driver: assume parameter "source" to describe the resource to run
  public void init() {
    initDriver();
    initGUI();
    }

  protected void initDriver(){
    driver = new AppletDriver(this);
    boolean debug = false;
    String useDebug = getParameter(PARAM_DEBUG);
    if (useDebug != null && (useDebug.equalsIgnoreCase("true") || useDebug.equalsIgnoreCase("yes")))
      debug = true;
    driver.setDebug(debug);
    }
  
  protected void initGUI() {
      try {
      // First see if raw definition is specified by the expr parameter
      // as a Mathematica expression
      String content = getParameter(PARAM_EXPR);
      if (content != null) {
        appletObject = driver.loadExpressionContent(content);
        return;
        }
        
      // Next see if raw definition is specified by the xml parameter
      // as a GUI_XMLFORMAT definition
      content = getParameter(PARAM_XML);
      if (content != null) {
        appletObject = driver.loadContent(content);
        return;
        }
        
      // Fallback to see if external resource path is specified by src parameter
      content = getParameter(PARAM_SRC);
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
      driver.execute(appletObject, AppletDriver.RELEASE_ONCLOSE, true);
      }
    else
      add(new Label("Applet parameters did not resolve to a valid GUIKit definition."));
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
