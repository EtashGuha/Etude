/*
 * @(#)GUIKitJTextPane.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.Component;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

import javax.swing.JTextPane;
import javax.swing.text.StyledDocument;

import com.oculustech.layout.OculusLayoutInfo;
import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.MathematicaEngineUtils;
import com.wolfram.guikit.GUIKitEnvironment;
import com.wolfram.guikit.util.FileUtils;

/**
 * GUIKitJTextPane extends JTextPane
 */
public class GUIKitJTextPane extends JTextPane implements OculusLayoutInfo {
 
    private static final long serialVersionUID = -1287986675456788648L;
    
	private MathematicaBSFEngine engine = null;
	private URL contextURL = null;
	private URL pathURL = null;
  
	/**
	 * Creates an uninitialized image icon.
	 */
	public GUIKitJTextPane() {
		super();
		}

  public GUIKitJTextPane(StyledDocument doc) {
    super(doc);
    }
    
  // methods of the OculusLayoutInfo interface to specify default stretching
  public int getXPreference() {return OculusLayoutInfo.CAN_BE_STRETCHED;}
  public int getYPreference() {return OculusLayoutInfo.NO_STRETCH;}
  public Component getSameHeightAs() {return null;}
  public Component getSameWidthAs() {return null;}
  
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

  public Object getPath() {
    if (pathURL != null) return pathURL.toExternalForm();
    return null;
    }
  public void setPath(Object path) {
    if (path != null) {
      if (path instanceof String) {
        pathURL = MathematicaEngineUtils.getMathematicaURL(
           GUIKitEnvironment.RESOLVE_FUNCTION, contextURL, (String)path, engine, null);
        }
      else if (path instanceof URL)
        pathURL = (URL)path;
        
      if (pathURL != null) {
        try {
          setPage(pathURL);
          }
        catch (IOException ex){}
        }
      }
    }
    
  public void setPage(String url) throws IOException {
    setPage(MathematicaEngineUtils.getMathematicaURL(
      GUIKitEnvironment.RESOLVE_FUNCTION, contextURL, url, engine, null));
    }
    
  public void setPage(URL page) throws IOException {
    super.setPage(page);
    }
  
  }
