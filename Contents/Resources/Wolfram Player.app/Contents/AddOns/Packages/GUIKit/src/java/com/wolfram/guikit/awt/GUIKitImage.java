/*
 * @(#)GUIKitImage.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.awt;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.ImageObserver;
import java.awt.image.ImageProducer;
import java.net.URL;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.guikit.swing.GUIKitImageIcon;

/**
 * GUIKitImage extends Image and exists for
 * convenience of defining an image based on the path or data properties
 * TODO this needs worked on because it currently does not render
 * properly in UIs. It may be that code expects a BufferedImage subclass
 *  and not just an Image interface
 */
public class GUIKitImage extends Image {
 
	private GUIKitImageIcon imageIcon;
    
	/**
	 * Creates an uninitialized image icon.
	 */
	public GUIKitImage() {
    imageIcon = new GUIKitImageIcon();
		}
		
  public void setEngine(MathematicaBSFEngine e) {
  	imageIcon.setEngine(e);
  	}
  	
  public String getContextPath() {
    return imageIcon.getContextPath();
    }
  public void setContextPath(String path) {
    imageIcon.setContextPath(path);
    }
    
  public URL getContextURL() {return imageIcon.getContextURL();}
  public void setContextURL(URL u) {
    imageIcon.setContextURL(u);
    }
  	
  public void setData(Object data) {
    imageIcon.setData(data);
    }
  
  public Object getPath() {
    return imageIcon.getPath();
    }
	public void setPath(Object path) {
		imageIcon.setPath(path);
		}
    
  // Abstract methods inherited from Image
  
  public int getWidth(ImageObserver observer) {
    if (imageIcon.getImage() != null)
      return imageIcon.getImage().getWidth(observer);
    return -1;
    }

  public int getHeight(ImageObserver observer) {
    if (imageIcon.getImage() != null)
      return imageIcon.getImage().getHeight(observer);
    return -1;
    }

  public ImageProducer getSource() {
    if (imageIcon.getImage() != null)
      return imageIcon.getImage().getSource();
    return null;
    }

  public Graphics getGraphics() {
    if (imageIcon.getImage() != null)
      return imageIcon.getImage().getGraphics();
    return null;
    }

  public Object getProperty(String name, ImageObserver observer) {
    if (imageIcon.getImage() != null)
      return imageIcon.getImage().getProperty(name, observer);
    return null;
    }
    
  public void flush() {
    if (imageIcon.getImage() != null)
      imageIcon.getImage().flush();
    }
   
  public Image getScaledInstance(int width, int height, int hints) {
    if (imageIcon.getImage() != null)
      return imageIcon.getImage().getScaledInstance(width, height, hints);
    return null;
    }
    
  }
