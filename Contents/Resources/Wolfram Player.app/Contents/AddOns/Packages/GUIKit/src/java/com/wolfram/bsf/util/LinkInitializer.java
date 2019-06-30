/*
 * @(#)LinkInitializer.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

/**
 * LinkInitializer
 */
public interface LinkInitializer {
   
  public void initializeLink(MathematicaBSFManager mathMgr, String contextName) throws MathematicaBSFException;
  
  }