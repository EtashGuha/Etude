/*
 * @(#)GUIKitImageIcon.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.net.MalformedURLException;
import java.net.URL;

import javax.swing.JLabel;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.guikit.util.FileUtils;

/**
 * GUIKitImageJLabel provides a simple
 * convenience JComponent for displaying an image
 */
public class GUIKitImageJLabel extends JLabel {
 
    private static final long serialVersionUID = -1277987975456777918L;
    
	private MathematicaBSFEngine engine = null;
	private URL contextURL = null;
	private URL pathURL = null;
    private URL disabledPathURL = null;
  
	/**
	 * Creates an uninitialized image icon.
	 */
	public GUIKitImageJLabel() {
		super("", null, CENTER);
		}
		
  public GUIKitImageJLabel(String text) {
    super(text, null, CENTER);
    }
    
  public void setEngine(MathematicaBSFEngine e) {
  	engine = e;
  	}
  	
  public String getContextPath() {
    if (contextURL != null) {
      return contextURL.toExternalForm();
      }
    return null;
    }
  public void setContextPath(String path) {
    if (path != null) {
      try {
        setContextURL(FileUtils.acceptedStringToURL(path));
        }
      catch (MalformedURLException me) {}
      }
    }
    
  public URL getContextURL() {return contextURL;}
  public void setContextURL(URL u) {
    contextURL = u;
    }
  	
  public void setData(Object data) {
    if (data != null) {
      GUIKitImageIcon icon = new GUIKitImageIcon();
      icon.setEngine(engine);
      icon.setContextURL(contextURL);
      icon.setData(data);
      if (icon.getImage() != null) {
        setIcon(icon);
        }
      }
    }
  
  public Object getPath() {
    if (pathURL != null) return pathURL.toExternalForm();
    return null;
    }
	public void setPath(Object path) {
		if (path != null) {
			GUIKitImageIcon icon = new GUIKitImageIcon();
      icon.setEngine(engine);
      icon.setContextURL(contextURL);
      icon.setPath(path);
      if (icon.getImage() != null) {
        setIcon(icon);
        pathURL = icon.getPathURL();
        }
			}
		}

  public void setDisabledData(Object data) {
    if (data != null) {
      GUIKitImageIcon icon = new GUIKitImageIcon();
      icon.setEngine(engine);
      icon.setContextURL(contextURL);
      icon.setData(data);
      if (icon.getImage() != null) {
        setDisabledIcon(icon);
        }
      }
    }
  
  public Object getDisabledPath() {
    if (disabledPathURL != null) return disabledPathURL.toExternalForm();
    return null;
    }
  public void setDisabledPath(Object path) {
    if (path != null) {
      GUIKitImageIcon icon = new GUIKitImageIcon();
      icon.setEngine(engine);
      icon.setContextURL(contextURL);
      icon.setPath(path);
      if (icon.getImage() != null) {
        setDisabledIcon(icon);
        disabledPathURL = icon.getPathURL();
        }
      }
    }
    
  }
