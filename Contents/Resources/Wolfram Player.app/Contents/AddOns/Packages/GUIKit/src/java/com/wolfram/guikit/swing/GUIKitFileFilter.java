/*
 * @(#)GUIKitFileFilter.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.io.File;
import javax.swing.filechooser.FileFilter;

/**
 * GUIKitFileFilter
 */
public class GUIKitFileFilter extends FileFilter {
 
  private String description = "Unnamed file filter";
  private String[] extensions = null;
  
	public GUIKitFileFilter() {
		super();
		}

  public boolean accept(File f) {
    if (f.isDirectory()) {
      return true;
      }
    String extension = getExtension(f);
    if (extension != null && extensions != null) {
      for (int i = 0; i < extensions.length; ++i) {
        if (extension.equalsIgnoreCase(extensions[i])) return true;
        }
      }
    return false;
    }   

  public String getDescription() {return description;}
  public void setDescription(String d) {description = d;}
    
  public String[] getExtensions() {return extensions;}
  public void setExtensions(String[] ex) {extensions = ex;}
  
  /*
   * Get the extension of a file.
   */
  public static String getExtension(File f) {
    String ext = null;
    String s = f.getName();
    int i = s.lastIndexOf('.');
    if (i > 0 &&  i < s.length() - 1) {
      ext = s.substring(i+1).toLowerCase();
      }
    return ext;
    }
    
  }
