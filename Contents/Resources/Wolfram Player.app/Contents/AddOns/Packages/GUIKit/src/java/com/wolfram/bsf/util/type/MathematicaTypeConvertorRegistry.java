/*
 * @(#)MathematicaTypeConvertorRegistry.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.bsf.util.type;

// BSF import switch
import org.apache.bsf.util.type.TypeConvertorRegistry;
import org.apache.bsf.util.type.TypeConvertor;
//

import com.wolfram.bsf.util.MathematicaBSFException;

import com.wolfram.jlink.Expr;

import java.math.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.text.NumberFormat;
import java.text.ParseException;

import java.awt.Color;

import java.util.Vector;
import java.lang.reflect.Array;

/**
 * MathematicaTypeConvertorRegistry
 */
public class MathematicaTypeConvertorRegistry extends TypeConvertorRegistry {
    
  public static MathematicaTypeConvertorRegistry typeConvertorRegistry = new MathematicaTypeConvertorRegistry();
  
  public TypedObject convert(TypedObject fromObj, Class toClass) throws MathematicaBSFException {
    if(fromObj.value == null || fromObj.type == null || toClass == fromObj.type || toClass.isAssignableFrom(fromObj.type)) {
      return TypedObjectFactory.create(toClass, fromObj.value);
      }
    else {
      TypeConvertor typeconvertor = lookup(fromObj.type, toClass);
      if (typeconvertor == null) {
        
        // Here unlike registered convertors we can handle some cases
        // Namely Object[] to another array type
        Class compClass = toClass.getComponentType();
        if (fromObj.type == Object[].class && compClass != null) {
          Object[] objs = (Object[])fromObj.value;
          Object arr = Array.newInstance(compClass, objs.length);
          try {
            for (int i = 0; i < objs.length; ++i) {
              Array.set(arr, i, convertAsObject(objs[i].getClass(), objs[i], compClass));
              }
            return TypedObjectFactory.create(toClass, arr);
            }
          catch (MathematicaBSFException me) {}
          } 
        
        throw new MathematicaBSFException(MathematicaBSFException.REASON_UNKNOWN_TYPECVTOR, 
            "unable to find a type converter from " + fromObj.type + " to " + toClass );
        }
      return TypedObjectFactory.create(toClass, typeconvertor.convert(fromObj.type, toClass, fromObj.value));
      }
    }
  
  public Object convertAsObject(TypedObject fromObj, Class toClass) throws MathematicaBSFException {
    return convertAsObject(fromObj.type, fromObj.value, toClass);
    }
    
  public Object convertAsObject(Class fromClass, Object fromObj, Class toClass) throws MathematicaBSFException {
    if(fromObj == null || fromClass == null || toClass == fromClass || toClass.isAssignableFrom(fromClass)) {
      return fromObj;
      }
    else {
      TypeConvertor typeconvertor = lookup(fromClass, toClass);
      if (typeconvertor == null) {
        
        // Here unlike registered convertors we can handle some cases
        // Namely Object[] to another array type
        Class compClass = toClass.getComponentType();
        if (fromClass == Object[].class && compClass != null) {
          Object[] objs = (Object[])fromObj;
          Object arr = Array.newInstance(compClass, objs.length);
          try {
            for (int i = 0; i < objs.length; ++i) {
              Array.set(arr, i, convertAsObject(objs[i].getClass(), objs[i], compClass));
              }
            return arr;
            }
          catch (MathematicaBSFException me) {}
          } 
          
        throw new MathematicaBSFException(MathematicaBSFException.REASON_UNKNOWN_TYPECVTOR, 
            "unable to find a type converter from " + fromClass + " to " + toClass );
        }
      return typeconvertor.convert(fromClass, toClass, fromObj);
      }
    }
    
  public MathematicaTypeConvertorRegistry () {
    super();

    // narrowing supported in J/Link for Double down to Float
    TypeConvertor tc = new TypeConvertor () {
      public Object convert (Class from, Class to, Object obj) {
        return new Float( ((Double)obj).floatValue());
      }
      public String getCodeGenString() {return "";}
    };
    register (Double.class, float.class, tc);
    register (Double.class, Float.class, tc);
    register (double.class, float.class, tc);
    register (double.class, Float.class, tc);
    
    // boolean to Object
    tc = new TypeConvertor () {
      public Object convert (Class from, Class to, Object obj) {
        return (Boolean)obj;
      }
      public String getCodeGenString() {return "";}
    };
    register(boolean.class, Object.class, tc);
    register(Boolean.class, Object.class, tc);
    
     // char to int 
    tc = new TypeConvertor () {
      public Object convert (Class from, Class to, Object obj) {
        return new Integer( (int)((Character)obj).charValue());
      }
      public String getCodeGenString() {return "";}
    };
    register (char.class, int.class, tc);
    register (char.class, Integer.class, tc);
    register (Character.class, int.class, tc);
    register (Character.class, Integer.class, tc);
    
    
    // char to String 
    tc = new TypeConvertor () {
      public Object convert (Class from, Class to, Object obj) {
        return new String( new char[]{((Character)obj).charValue()});
      }
      public String getCodeGenString() {return "";}
    };
    register (char.class, String.class, tc);
    register (Character.class, String.class, tc);
    
    // String to Character 
    tc = new TypeConvertor () {
      public Object convert (Class from, Class to, Object obj) {
        return new Character(((String)obj).charAt(0));
      }
      public String getCodeGenString() {return "";}
    };
    register (String.class, char.class, tc);
    register (String.class, Character.class, tc);
    
    TypeConvertor convertor = null;

    convertor = new ExprTypeConvertor();
    
    register(boolean.class, Expr.class, convertor);
    register(Boolean.class, Expr.class, convertor);
    register(int.class, Expr.class, convertor);
    register(Integer.class, Expr.class, convertor);
    register(long.class, Expr.class, convertor);
    register(Long.class, Expr.class, convertor);
    register(double.class, Expr.class, convertor);
    register(Double.class, Expr.class, convertor);
    register(float.class, Expr.class, convertor);
    register(Float.class, Expr.class, convertor);
    register(BigDecimal.class, Expr.class, convertor);
    register(BigInteger.class, Expr.class, convertor);
    register(String.class, Expr.class, convertor);
    register(Color.class, Expr.class, convertor);
		
		register(Expr.class, boolean.class, convertor);
		register(Expr.class, Boolean.class, convertor);
		register(Expr.class, int.class, convertor);
		register(Expr.class, Integer.class, convertor);
    register(Expr.class, byte.class, convertor);
    register(Expr.class, Byte.class, convertor);
		register(Expr.class, long.class, convertor);
		register(Expr.class, Long.class, convertor);
		register(Expr.class, double.class, convertor);
		register(Expr.class, Double.class, convertor);
    register(Expr.class, float.class, convertor);
    register(Expr.class, Float.class, convertor);
		register(Expr.class, BigDecimal.class, convertor);
		register(Expr.class, BigInteger.class, convertor);
		register(Expr.class, String.class, convertor);
    register(Expr.class, Color.class, convertor);
    register(Expr.class, Object[][].class, convertor);
		register(Expr.class, Object[].class, convertor);
		register(Expr.class, Object.class, convertor);
		register(Expr.class, Vector.class, convertor);
    
    register(Expr.class, byte[].class, convertor);
    register(Expr.class, Byte[].class, convertor);
    register(Expr.class, int[].class, convertor);
    register(Expr.class, Integer[].class, convertor);
    register(Expr.class, double[].class, convertor);
    register(Expr.class, Double[].class, convertor);
    register(Expr.class, String[].class, convertor);
    
    // A no op convertor to support registering to Number
    tc = new TypeConvertor() {
      public Object convert (Class from, Class to, Object obj) {
        return obj;}
      public String getCodeGenString() {return "";}
    };
    
    register(Byte.class, Number.class, tc);
    register(byte.class, Number.class, tc);
    register(Short.class, Number.class, tc);
    register(short.class, Number.class, tc);
    register(Integer.class, Number.class, tc);
    register(int.class, Number.class, tc);
    register(Long.class, Number.class, tc);
    register(long.class, Number.class, tc);
    register(Float.class, Number.class, tc);
    register(float.class, Number.class, tc);
    register(Double.class, Number.class, tc);
    register(double.class, Number.class, tc);
    
    register(String.class, URL.class, 
     new TypeConvertor() {
      public Object convert (Class from, Class to, Object obj) {
        URL u = null;
        try {
          u = new URL((String)obj);
        }
        catch (MalformedURLException e) {}
        return u;}
      public String getCodeGenString() {return "";}
      });
      
    register(String.class, Number.class, 
     new TypeConvertor() {
      public Object convert (Class from, Class to, Object obj) {
        Number num = null;
        try {
          num = NumberFormat.getInstance().parse((String)obj);
        }
        catch (ParseException pe) {}
        return num;}
      public String getCodeGenString() {return "";}
      });
    
			//	A primitive array to Object array convertor
		 tc = new TypeConvertor() {
			 public Object convert (Class from, Class to, Object obj) {
				 if (from == int[].class) {
				 		int[] origArr = (int[])obj;
				 	  Object[] objArr = new Object[origArr.length];
				 	  for (int i = 0; i < origArr.length; ++i)
							objArr[i] = new Integer(origArr[i]);
				 		return objArr;
				 		}
				 else if (from == float[].class) {
					  float[] origArr = (float[])obj;
						Object[] objArr = new Object[origArr.length];
						for (int i = 0; i < origArr.length; ++i)
							objArr[i] = new Float(origArr[i]);
						return objArr;
						}
				else if (from == double[].class) {
					  double[] origArr = (double[])obj;
						Object[] objArr = new Object[origArr.length];
						for (int i = 0; i < origArr.length; ++i)
							objArr[i] = new Double(origArr[i]);
						return objArr;
						}
        else if (from == byte[].class) {
            byte[] origArr = (byte[])obj;
            Object[] objArr = new Object[origArr.length];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = new Byte(origArr[i]);
            return objArr;
            }
				else if (from == char[].class) {
					  char[] origArr = (char[])obj;
						Object[] objArr = new Object[origArr.length];
						for (int i = 0; i < origArr.length; ++i)
							objArr[i] = new Character(origArr[i]);
						return objArr;
            }
        else if (from == boolean[].class) {
            boolean[] origArr = (boolean[])obj;
            Object[] objArr = new Object[origArr.length];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = origArr[i] ? Boolean.TRUE : Boolean.FALSE;
            return objArr;
            }
        else if (from == int[][].class) {
            int[][] origArr = (int[][])obj;
            Object[][] objArr = new Object[origArr.length][];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = (Object[])convert(int[].class, Object[].class, origArr[i]);
            return objArr;
            }
        else if (from == float[][].class) {
            float[][] origArr = (float[][])obj;
            Object[][] objArr = new Object[origArr.length][];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = (Object[])convert(float[].class, Object[].class, origArr[i]);
            return objArr;
            }
        else if (from == double[][].class) {
            double[][] origArr = (double[][])obj;
            Object[][] objArr = new Object[origArr.length][];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = (Object[])convert(double[].class, Object[].class, origArr[i]);
            return objArr;
            }
        else if (from == byte[][].class) {
            byte[][] origArr = (byte[][])obj;
            Object[][] objArr = new Object[origArr.length][];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = (Object[])convert(byte[].class, Object[].class, origArr[i]);
            return objArr;
            }
        else if (from == char[][].class) {
            char[][] origArr = (char[][])obj;
            Object[][] objArr = new Object[origArr.length][];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = (Object[])convert(char[].class, Object[].class, origArr[i]);
            return objArr;
            }
         else if (from == boolean[][].class) {
            boolean[][] origArr = (boolean[][])obj;
            Object[][] objArr = new Object[origArr.length][];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = (Object[])convert(boolean[].class, Object[].class, origArr[i]);
            return objArr;
            }
				 return obj;
				 }
			 public String getCodeGenString() {return "";}
		 };
    
		register(int[].class, Object[].class, tc);
		register(float[].class, Object[].class, tc);
		register(double[].class, Object[].class, tc);
		register(char[].class, Object[].class, tc);
		register(boolean[].class, Object[].class, tc);
    register(byte[].class, Object[].class, tc);
        
    register(int[][].class, Object[][].class, tc);
    register(float[][].class, Object[][].class, tc);
    register(double[][].class, Object[][].class, tc);
    register(char[][].class, Object[][].class, tc);
    register(boolean[][].class, Object[][].class, tc);
    register(byte[][].class, Object[][].class, tc);
    
    
    //  A common int[] to byte[] downcasting especially
    // because of J/Links use of int[] for conversion from Mathematica
     tc = new TypeConvertor() {
       public Object convert (Class from, Class to, Object obj) {
         if (from == int[].class) {
            int[] origArr = (int[])obj;
            byte[] objArr = new byte[origArr.length];
            for (int i = 0; i < origArr.length; ++i)
              objArr[i] = (byte)origArr[i];
            return objArr;
            }
         return obj;
         }
       public String getCodeGenString() {return "";}
     };
     
    register(int[].class, byte[].class, tc);
     
    //  A primitive array converter
     tc = new TypeConvertor() {
       public Object convert (Class from, Class to, Object obj) {
         if (from == int[].class || from == float[].class || 
             from == double[].class || from == char[].class) {
            Object res = null;
            try {
              res = convertAsObject(from, obj, Object[].class);
              }
            catch (MathematicaBSFException me) {}
            return convert( Object[].class, to, res);
            }
         else if (from == Object[].class) {
            if  (to == Vector.class) {
              Object[] origArr = (Object[])obj;
              Vector vec = new Vector(origArr.length);
              for (int i = 0; i < origArr.length; ++i)
                vec.addElement(origArr[i]);
              return vec;
              }
            else if (to == String[].class) {
              Object[] origArr = (Object[])obj;
              String[] strArr = new String[origArr.length];
              for (int i = 0; i < origArr.length; ++i)
                strArr[i] = origArr[i].toString();
              return strArr;
              }
						else if (to == byte[].class) {
							Object[] origArr = (Object[])obj;
							byte[] byteArr = new byte[origArr.length];
							for (int i = 0; i < origArr.length; ++i) {
								Object o = origArr[i];
								if (o == null) continue;
								if (o instanceof Number) byteArr[i] = ((Number)origArr[i]).byteValue();
								}
							return byteArr;
							}
            }
          else if (from == String[].class) {
            String[] origArr = (String[])obj;
            Vector vec = new Vector(origArr.length);
            for (int i = 0; i < origArr.length; ++i)
              vec.addElement(origArr[i]);
            return vec;
            }
         return obj;
         }
       public String getCodeGenString() {return "";}
     };
    
    register(int[].class, Vector.class, tc);
    register(float[].class, Vector.class, tc);
    register(double[].class, Vector.class, tc);
    register(char[].class, Vector.class, tc);
    register(Object[].class, Vector.class, tc);
    register(Object[].class, String[].class, tc);
		register(Object[].class, byte[].class, tc);
    register(String[].class, Vector.class, tc);
    }
	
  }

