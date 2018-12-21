using Test
using ParametersType
@parameter a π "parameter a"
@parameter b π "parameter b"
@parameter c 2a+b "parameter c"
@parameter d 2//3 "parameter d"

module Foo
  using ParametersType
  @parameter a1 π "parameter a1"
  @parameter b1 π "parameter b1"
  @parameter c1 2a1+b1 "parameter c1"
  @exportparameter a2 π "parameter a2"
  @exportparameter b2 π "parameter b2"
  @exportparameter c2 2a2+b2 "parameter c2"
end

using Main.Foo

function main()
  println()
  println("Testing: ",relpath(@__FILE__))
  @testset begin
    @test a==a
    @test a==b
    @test a!=c
    @test !(a==c)
    @test !(a!=a)

    @test ParametersType.getval(a) == π
    @test a == π
    @test ParametersType.getval(a) != Float64(π)
    @test ParametersType.getval(d) == 2//3
    @test d == 2//3
    @test ParametersType.getval(d) != Float64(2//3)

    @test a<=b
    @test a<=a

    @test !(b<a)
    @test !(a<a)

    @test  a<c
    @test  a<=c
    @test !(a<a)
    @test !(a>=c)
    @test !(a>c)

    @test !@isdefined a1
    @test !@isdefined b1
    @test !@isdefined c1

    @test @isdefined a2
    @test @isdefined b2
    @test @isdefined c2

    @test try
      Foo.a1
      Foo.b1
      Foo.c1
      true
    catch
      false
    end
  end
end

main()
