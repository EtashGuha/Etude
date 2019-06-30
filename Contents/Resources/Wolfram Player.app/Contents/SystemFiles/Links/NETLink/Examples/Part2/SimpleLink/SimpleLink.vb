''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'
'  SimpleLink.vb
'
'  A very simple .NET/Link example program demonstrating various methods from the
'  IKernelLink interface.
'
'  To compile this program, see the ReadMe.html file that accompanies it.
'
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Imports System
Imports Wolfram.NETLink

Public Class SimpleLink

    Public Shared Sub Main(ByVal args As String())

        ' This launches the Mathematica kernel:
        Dim ml As IKernelLink = MathLinkFactory.CreateKernelLink()

        ' Discard the initial InputNamePacket the kernel will send when launched.
         ml.WaitAndDiscardAnswer()

        ' Now compute 2+2 in several different ways.

        ' The easiest way. Send the computation as a string and get the result in a single call:
        Dim result As String = ml.EvaluateToOutputForm("2+2", 0)
        Console.WriteLine("2 + 2 = " & result)

        ' Use Evaluate() instead of EvaluateToXXX() if you want to read the result
        ' as a native type instead of a string.
        ml.Evaluate("2+2")
        ml.WaitForAnswer()
        Dim intResult As Integer = ml.GetInteger()
        Console.WriteLine("2 + 2 = " & intResult)

        ' You can also get down to the metal by using methods from IMathLink:
        ml.PutFunction("EvaluatePacket", 1)
        ml.PutFunction("Plus", 2)
        ml.Put(2)
        ml.Put(2)
        ml.EndPacket()
        ml.WaitForAnswer()
        intResult = ml.GetInteger()
        Console.WriteLine("2 + 2 = " & intResult)

        'Always Close when done:
        ml.Close()
     End Sub

End Class
