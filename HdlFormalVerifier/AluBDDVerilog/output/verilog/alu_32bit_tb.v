//==============================================================================
// Testbench: alu_32bit
// Generated: 2025-11-27 00:50:58
// Generator: testbench_generator.py (deterministic, no LLM)
//
// DUT Module: alu_32bit
// Bitwidth: 32-bit
// Test cases: 20
// Number format: decimal
//==============================================================================

`timescale 1ns / 1ps

module alu_32bit_tb;

    //--------------------------------------------------------------------------
    // Test Signals
    //--------------------------------------------------------------------------
    reg  [31:0] a;
    reg  [31:0] b;
    reg  [3:0] opcode;
    wire [31:0] result;
    wire zero, carry, overflow, negative;

    // Test counters
    integer passed = 0;
    integer failed = 0;
    integer total = 0;

    //--------------------------------------------------------------------------
    // Device Under Test (DUT)
    //--------------------------------------------------------------------------
    alu_32bit uut (
        .a(a),
        .b(b),
        .opcode(opcode),
        .result(result),
        .zero(zero),
        .carry(carry),
        .overflow(overflow),
        .negative(negative)
    );

    //--------------------------------------------------------------------------
    // Test Sequence
    //--------------------------------------------------------------------------
    initial begin
        $display("========================================");
        $display("Testbench: alu_32bit");
        $display("DUT: alu_32bit");
        $display("========================================");
        $display("");

        // Test 1: ADD
        a = 32'd10;
        b = 32'd5;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd15);
            failed = failed + 1;
        end

        // Test 2: ADD
        a = 32'd118;
        b = 32'd118;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd236) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd236);
            failed = failed + 1;
        end

        // Test 3: ADD
        a = 32'd92;
        b = 32'd0;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd92) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd92);
            failed = failed + 1;
        end

        // Test 4: ADD
        a = 32'd1398439431;
        b = 32'd1805366006;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd3203805437) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd3203805437);
            failed = failed + 1;
        end

        // Test 5: ADD
        a = 32'd15785585;
        b = 32'd1214608580;
        opcode = 4'b0000;
        #10;
        total = total + 1;
        $display("Test %0d: %d ADD %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd1230394165) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd1230394165);
            failed = failed + 1;
        end

        // Test 6: SUB
        a = 32'd10;
        b = 32'd5;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd5) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd5);
            failed = failed + 1;
        end

        // Test 7: SUB
        a = 32'd400;
        b = 32'd400;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd0);
            failed = failed + 1;
        end

        // Test 8: SUB
        a = 32'd34;
        b = 32'd0;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd34) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd34);
            failed = failed + 1;
        end

        // Test 9: SUB
        a = 32'd247269718;
        b = 32'd1766346966;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd2775890048) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd2775890048);
            failed = failed + 1;
        end

        // Test 10: SUB
        a = 32'd2063751945;
        b = 32'd984423246;
        opcode = 4'b0001;
        #10;
        total = total + 1;
        $display("Test %0d: %d SUB %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd1079328699) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd1079328699);
            failed = failed + 1;
        end

        // Test 11: AND
        a = 32'd10;
        b = 32'd5;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd0);
            failed = failed + 1;
        end

        // Test 12: AND
        a = 32'd628;
        b = 32'd628;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd628) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd628);
            failed = failed + 1;
        end

        // Test 13: AND
        a = 32'd46;
        b = 32'd0;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd0) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd0);
            failed = failed + 1;
        end

        // Test 14: AND
        a = 32'd1039783140;
        b = 32'd90182474;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd90181696) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd90181696);
            failed = failed + 1;
        end

        // Test 15: AND
        a = 32'd1347050235;
        b = 32'd1472843118;
        opcode = 4'b0010;
        #10;
        total = total + 1;
        $display("Test %0d: %d AND %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd1346914410) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd1346914410);
            failed = failed + 1;
        end

        // Test 16: OR
        a = 32'd10;
        b = 32'd5;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd15) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd15);
            failed = failed + 1;
        end

        // Test 17: OR
        a = 32'd580;
        b = 32'd580;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd580) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd580);
            failed = failed + 1;
        end

        // Test 18: OR
        a = 32'd27;
        b = 32'd0;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd27) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd27);
            failed = failed + 1;
        end

        // Test 19: OR
        a = 32'd721662266;
        b = 32'd985956352;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd1002942778) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd1002942778);
            failed = failed + 1;
        end

        // Test 20: OR
        a = 32'd2061291246;
        b = 32'd147068871;
        opcode = 4'b0011;
        #10;
        total = total + 1;
        $display("Test %0d: %d OR %d = %d (Z=%b C=%b O=%b N=%b)",
                 total, a, b, result, zero, carry, overflow, negative);
        if (result == 32'd2061295599) begin
            $display("  [PASS] Result correct");
            passed = passed + 1;
        end else begin
            $display("  [FAIL] Expected: %d", 32'd2061295599);
            failed = failed + 1;
        end

        $display("");
        $display("========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total:  %0d", total);
        $display("Passed: %0d", passed);
        $display("Failed: %0d", failed);
        if (failed == 0)
            $display("[SUCCESS] All tests passed!");
        else
            $display("[FAILURE] Some tests failed!");
        $display("========================================");

        #10;
        $finish;
    end

    //--------------------------------------------------------------------------
    // VCD Dump for Waveform Viewing
    //--------------------------------------------------------------------------
    initial begin
        $dumpfile("alu_32bit_tb.vcd");
        $dumpvars(0, alu_32bit_tb);
    end

endmodule

//==============================================================================
// End of Testbench
//==============================================================================