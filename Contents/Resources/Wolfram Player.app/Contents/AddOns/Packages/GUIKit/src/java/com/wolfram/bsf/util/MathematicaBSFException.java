/*
 * @(#)MathematicaBSFException.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util;

// BSF import switch
import org.apache.bsf.BSFException;
//

/**
 * MathematicaBSFException
 */
public class MathematicaBSFException extends BSFException {
 
  private static final long serialVersionUID = -1587987975456388048L;
  
  public static int REASON_UNKNOWN_TYPECVTOR = 12;
  
  public MathematicaBSFException(int reason, String msg) {
    super(reason, msg);
    }

  public MathematicaBSFException(int reason, String msg, Throwable t) {
    super(reason, msg, t);
    }

  public MathematicaBSFException(String msg) {
    super(msg);
    }
}
