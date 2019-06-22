/*
 * @(#)GUIKitImageIcon.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.Image;
import java.awt.Toolkit;
import java.net.MalformedURLException;
import java.net.URL;

import javax.swing.ImageIcon;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.bsf.util.MathematicaEngineUtils;
import com.wolfram.guikit.GUIKitEnvironment;
import com.wolfram.guikit.GUIKitUtils;
import com.wolfram.guikit.util.FileUtils;

/**
 * GUIKitImageIcon extends ImageIcon
 */
public class GUIKitImageIcon extends ImageIcon {
 
    private static final long serialVersionUID = -1237987975456733948L;
    
	private MathematicaBSFEngine engine = null;
	private URL contextURL = null;
	private URL pathURL = null;
  
	/**
	 * Creates an uninitialized image icon.
	 */
	public GUIKitImageIcon() {
		super();
		}
		
	public GUIKitImageIcon(String filename) {
		super(filename);
		}
	
	public GUIKitImageIcon(URL location) {
		super(location);
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
      Image image = GUIKitUtils.createImage(data);
      if (image != null) {
        setImage(image);
        } 
      }
    }
  
  public URL getPathURL() {return pathURL;}
  
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
				Image image = Toolkit.getDefaultToolkit().getImage(pathURL);
				if (image != null) {
					setImage(image);
					} 
		  	}
			}
		}


  }
