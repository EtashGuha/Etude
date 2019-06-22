/*
 * @(#)FileUtils.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.util;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * FileUtils
 */
public class FileUtils {

  public static URL acceptedStringToURL(String path) throws MalformedURLException {
    if (path.startsWith("http:") || path.startsWith("https:") || path.startsWith("file:"))
      return new URL(path);
    return new URL("file", "", path);
    }
  
  public static String readTextFile(File f) {
    if (f != null) {
      FileReader fr = null;
      try {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        fr = new FileReader(f);
        BufferedReader in = new BufferedReader(fr);
        String tempLine;

        while ((tempLine = in.readLine()) != null) {
          pw.println(tempLine);
          }
        pw.flush();
        return sw.toString();
        }
      catch (IOException e) {
        e.printStackTrace();
        }
      finally {
        try {
          if (fr != null) fr.close();
          }
        catch (Exception ex) {}
        }
      }
    return null;
    }

  public static void writeTextFile(String str, File f) {
    if (f != null) {
      FileWriter fw = null;
      try {
        fw = new FileWriter(f);
        PrintWriter pw = new PrintWriter(fw);
        BufferedReader in = new BufferedReader(new StringReader(str));
        String tempLine;

        while ((tempLine = in.readLine()) != null) {
          pw.println(tempLine);
          }
        pw.flush();
        }
      catch (IOException e) {
        e.printStackTrace();
        }
      finally {
        try {
          if (fw != null) fw.close();
          }
        catch (Exception ex) {}
        }
      }
    }

  }
