/*
 * @(#)GUIKitApplication.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.app;

import java.io.*;
import java.net.*;

/**
 * GUIKitApplication is an implementation of a GUIKit ruuntime as an application
 *
 * Note this class sets a handler to call System.exit(0) and so should only
 * be used as a main class to a standalone application and not used within a JLink VM
 * shared with other code.
 *
 * Will it ever make sense for this to support/use the modal API, probably not
 */
public class GUIKitApplication {

  private int activeCount = 0;

  public GUIKitApplication(String[] args) throws MalformedURLException, IOException {
    for (int i = 0; i < args.length; ++i) {
      ApplicationDriver driver = new ApplicationDriver(this);
      
      //TODO change this flag name once we have final package name
      if (System.getProperty("debug") != null)
        driver.setDebug(true);
      else driver.setDebug(false);
      
      try {
        driver.runFile(args[i]);
        activeCount++;
        }
      catch (Exception e) {
        e.printStackTrace();
        }
      }
    }

  public void finished() {
    System.exit(0);
    }

  public void driverFinished() {
    activeCount--;
    if(activeCount <= 0) {
      finished();
      }
    }

  public static void main(String[] args) {
    try {
      new GUIKitApplication(args);
      }
    catch (Exception e) {
      e.printStackTrace();
      }
    }

}
