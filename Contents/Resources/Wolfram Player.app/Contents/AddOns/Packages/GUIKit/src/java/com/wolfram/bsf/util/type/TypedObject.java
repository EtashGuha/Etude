/*
 * @(#)TypedObject.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.type;

// BSF import switch
import org.apache.bsf.util.Bean;
//

/**
 * TypedObject is a lightweight abstracted BSF Bean object
 */
public class TypedObject extends Bean {

  public static final TypedObject TYPED_NULL = new TypedObject(Object.class, null);
  
  public TypedObject(Class type, Object value) {
    super(type, value);
    }

}

